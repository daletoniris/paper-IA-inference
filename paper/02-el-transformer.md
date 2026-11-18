# Parte 2: El Transformer -- La Arquitectura que Cambio Todo

## 2.1 El problema de las redes recurrentes

Antes de los Transformers, el estado del arte para procesar texto eran
las redes recurrentes (RNN) y su variante LSTM. Procesaban texto de
forma secuencial: una palabra a la vez, de izquierda a derecha.

Problemas:
- **Lento**: no se puede paralelizar (cada paso depende del anterior).
- **Memoria limitada**: despues de muchas palabras, "olvida" las primeras.
- **Gradientes que desaparecen**: el entrenamiento se degrada en secuencias largas.

## 2.2 "Attention Is All You Need" (2017)

En junio de 2017, Vaswani et al. (Google) publicaron un paper que cambio
todo: propusieron reemplazar la recurrencia con un mecanismo llamado
**self-attention** que permite ver TODA la secuencia al mismo tiempo.

Resultado: entrenamiento masivamente paralelo, contexto sin limite practico,
y el nacimiento de GPT, BERT, T5, y todo lo que vino despues.

## 2.3 Tokenizacion: de texto a numeros

Un modelo no entiende texto. Solo entiende numeros.

La tokenizacion convierte texto en una secuencia de IDs numericos:

```
"Hola mundo"  ->  tokenizar  ->  [15339, 29871, 15426]
```

Los tokenizadores modernos (BPE, SentencePiece) dividen el texto en
subpalabras, no en palabras completas:

```
"cybersecurity"  ->  ["cyber", "security"]  ->  [8941, 6400]
"unhackeable"    ->  ["un", "hack", "eable"] ->  [348, 29012, 3362]
```

Esto permite manejar palabras nuevas o raras sin un vocabulario infinito.

```python
# Ejemplo conceptual de tokenizacion simple
vocabulario = {
    "hola": 0, "mundo": 1, "que": 2, "tal": 3,
    "como": 4, "estas": 5, "<pad>": 6, "<unk>": 7,
}

def tokenizar(texto):
    """Convierte texto en IDs numericos."""
    tokens = texto.lower().split()
    return [vocabulario.get(t, vocabulario["<unk>"]) for t in tokens]

# "hola mundo" -> [0, 1]
# "hola robot" -> [0, 7]  (robot no esta en el vocabulario -> <unk>)
```


## 2.4 Embeddings: de numeros a vectores con significado

Un token ID es solo un indice. No tiene significado matematico.
El embedding convierte cada ID en un vector de numeros reales que
captura relaciones semanticas.

```
token "rey"    ->  ID 4521  ->  embedding [0.23, -0.45, 0.78, 0.12, ...]
token "reina"  ->  ID 7834  ->  embedding [0.25, -0.42, 0.76, 0.89, ...]
token "gato"   ->  ID 2901  ->  embedding [0.91,  0.33, -0.22, 0.01, ...]
```

"Rey" y "reina" tienen vectores similares. "Gato" tiene un vector diferente.
El modelo aprende estos vectores durante el entrenamiento.

La famosa relacion: `rey - hombre + mujer ~ reina` funciona porque
las operaciones aritmeticas en el espacio de embeddings reflejan
relaciones semanticas.

```python
import numpy as np

# Ejemplo simplificado: embedding de dimension 4
# En GPT-4 cada token tiene un embedding de ~12,000 dimensiones.
embeddings = {
    "rey":   np.array([0.9,  0.1,  0.8,  0.2]),
    "reina": np.array([0.9,  0.1,  0.8,  0.9]),   # Similar a rey, diferente en genero
    "hombre":np.array([0.1,  0.1,  0.1,  0.2]),
    "mujer": np.array([0.1,  0.1,  0.1,  0.9]),   # Similar a hombre, diferente en genero
}

# rey - hombre + mujer = ?
resultado = embeddings["rey"] - embeddings["hombre"] + embeddings["mujer"]
print(f"rey - hombre + mujer = {resultado}")
# Resultado cercano a [0.9, 0.1, 0.8, 0.9] = "reina"
```


## 2.5 Self-Attention: la pieza clave

Self-attention permite que cada token "mire" a todos los demas tokens
de la secuencia y decida cuales son relevantes para el.

Para la frase: "El gato se sento en la alfombra porque estaba cansado"
- La palabra "estaba" necesita saber que se refiere a "gato", no a "alfombra".
- Self-attention calcula esta relacion automaticamente.

**Mecanica:**

Cada token se transforma en tres vectores:
- **Q (Query)**: "que estoy buscando?"
- **K (Key)**: "que informacion tengo?"
- **V (Value)**: "que valor aporto?"

La atencion se calcula como:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

1. Q * K^T: "cuanto se parece mi query a cada key?" (scores de atencion)
2. / sqrt(d_k): normalizar para evitar que los scores exploten
3. softmax: convertir en probabilidades (suman 1)
4. * V: ponderar los valores por la atencion calculada

```python
import numpy as np

def self_attention(Q, K, V):
    """
    Self-attention: el mecanismo central del Transformer.

    Q: queries  (seq_len, d_k)
    K: keys     (seq_len, d_k)
    V: values   (seq_len, d_v)

    Retorna: salida ponderada por atencion (seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # Paso 1: calcular scores de atencion
    scores = Q @ K.T / np.sqrt(d_k)

    # Paso 2: convertir a probabilidades con softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Paso 3: ponderar los valores
    output = attention_weights @ V

    return output, attention_weights


# Ejemplo: 3 tokens, dimension 4
np.random.seed(42)
seq_len = 3
d_model = 4

# Embeddings de 3 tokens
embeddings = np.random.randn(seq_len, d_model)

# Matrices de transformacion (aprendidas durante entrenamiento)
W_Q = np.random.randn(d_model, d_model) * 0.5
W_K = np.random.randn(d_model, d_model) * 0.5
W_V = np.random.randn(d_model, d_model) * 0.5

# Calcular Q, K, V
Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V

output, weights = self_attention(Q, K, V)

print("Pesos de atencion (cada fila muestra cuanto atiende un token a los demas):")
print(np.round(weights, 3))
print()
print("Salida (cada token es una combinacion ponderada de todos los valores):")
print(np.round(output, 3))
```


## 2.6 Multi-Head Attention

En vez de una sola atencion, el Transformer usa multiples "cabezas"
(heads) en paralelo. Cada cabeza aprende a prestar atencion a
diferentes tipos de relaciones:

- Head 1 podria aprender relaciones gramaticales (sujeto-verbo)
- Head 2 podria aprender relaciones semanticas (sinonimos)
- Head 3 podria aprender relaciones posicionales (palabras cercanas)

Los resultados de todas las cabezas se concatenan y se proyectan.

```python
def multi_head_attention(embeddings, num_heads, d_model):
    """
    Multi-head attention: multiples perspectivas simultaneas.
    """
    d_head = d_model // num_heads
    heads_output = []

    for h in range(num_heads):
        # Cada cabeza tiene sus propias matrices W_Q, W_K, W_V
        W_Q = np.random.randn(d_model, d_head) * 0.5
        W_K = np.random.randn(d_model, d_head) * 0.5
        W_V = np.random.randn(d_model, d_head) * 0.5

        Q = embeddings @ W_Q
        K = embeddings @ W_K
        V = embeddings @ W_V

        head_out, _ = self_attention(Q, K, V)
        heads_output.append(head_out)

    # Concatenar todas las cabezas
    concatenated = np.concatenate(heads_output, axis=-1)

    # Proyeccion final
    W_O = np.random.randn(d_model, d_model) * 0.5
    output = concatenated @ W_O

    return output
```


## 2.7 Positional Encoding

Self-attention no tiene nocion de orden. Para la atencion,
"el gato come pescado" y "pescado come el gato" son equivalentes.

El positional encoding agrega informacion de posicion a cada embedding:

```python
def positional_encoding(seq_len, d_model):
    """
    Genera vectores de posicion usando funciones seno/coseno.
    Cada posicion tiene un patron unico que el modelo puede aprender.
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return PE

# El embedding final de cada token es:
# embedding_final = embedding_token + positional_encoding
```


## 2.8 Feed-Forward + LayerNorm + Residual Connections

Despues de la atencion, cada token pasa por una red feed-forward
(dos capas lineales con activacion ReLU en el medio):

```python
def feed_forward(x, d_model, d_ff):
    """Red feed-forward de dos capas con ReLU."""
    W1 = np.random.randn(d_model, d_ff) * 0.5
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.5
    b2 = np.zeros(d_model)

    hidden = np.maximum(0, x @ W1 + b1)   # ReLU
    output = hidden @ W2 + b2
    return output
```

**Residual connections** (conexiones residuales) suman la entrada
original a la salida de cada sub-capa. Esto permite entrenar redes
muy profundas sin que los gradientes desaparezcan:

```
output = LayerNorm(x + Attention(x))
output = LayerNorm(output + FeedForward(output))
```


## 2.9 Codigo: Transformer decoder-only desde cero (PyTorch)

Este es un mini-transformer funcional que se puede entrenar en CPU.
Mismo principio que GPT, pero con ~100K parametros en vez de miles
de millones.

```python
#!/usr/bin/env python3
"""
mini_transformer.py -- Transformer decoder-only desde cero en PyTorch.
Entrenable en CPU. Misma arquitectura que GPT, escala minima.

Aprende a completar secuencias de texto de un dataset pequeno.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Self-attention con mascara causal (el token solo puede ver
    tokens anteriores, no futuros -- esto lo hace "autoregresivo").
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Proyecciones Q, K, V (una matriz por cada una)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Proyeccion de salida
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Calcular Q, K, V de una sola vez (eficiente)
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape para multi-head: (batch, heads, seq, d_head)
        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Mascara causal: impide ver tokens futuros
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Reshape y proyeccion de salida
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    """
    Un bloque del transformer: Attention + FFN + LayerNorm + Residual.
    GPT apila muchos de estos bloques (12, 24, 96, etc.).
    """

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                # Activacion GELU (usada en GPT)
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm (como en GPT-2+): normalizar ANTES de la sub-capa
        x = x + self.attention(self.norm1(x))   # Residual + Attention
        x = x + self.ffn(self.norm2(x))          # Residual + FFN
        return x


class MiniTransformer(nn.Module):
    """
    Transformer decoder-only completo.
    Misma arquitectura que GPT, pero en escala minima.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()

        # Embedding de tokens: cada token ID -> vector de d_model dimensiones
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: cada posicion -> vector de d_model dimensiones
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Apilar N bloques de transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Normalizacion final
        self.final_norm = nn.LayerNorm(d_model)

        # Cabeza de prediccion: embedding -> logits sobre vocabulario
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape

        # Embeddings = token embedding + position embedding
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        # Pasar por todos los bloques del transformer
        for block in self.blocks:
            x = block(x)

        # Normalizar y proyectar a vocabulario
        x = self.final_norm(x)
        logits = self.head(x)

        return logits

    def generar(self, inicio_ids, max_nuevos_tokens, temperatura=0.8):
        """
        Genera texto token por token (autoregresivo).
        Esto es EXACTAMENTE lo que hace GPT cuando te responde.
        """
        self.eval()
        ids = inicio_ids.clone()

        with torch.no_grad():
            for _ in range(max_nuevos_tokens):
                # Forward pass con toda la secuencia generada hasta ahora
                logits = self(ids)

                # Solo nos importa la prediccion del ULTIMO token
                logits_ultimo = logits[:, -1, :] / temperatura

                # Muestrear el siguiente token
                probs = F.softmax(logits_ultimo, dim=-1)
                siguiente = torch.multinomial(probs, 1)

                # Agregar al contexto
                ids = torch.cat([ids, siguiente], dim=-1)

        return ids


# ========================================================================
# ENTRENAMIENTO
# ========================================================================

# Dataset: frases cortas para entrenar el mini-transformer
textos = [
    "el agente ejecuta herramientas",
    "la seguridad esta en la arquitectura",
    "el modelo predice el siguiente token",
    "un transformer usa atencion para entender contexto",
    "los pesos son el conocimiento del modelo",
    "la inferencia usa los pesos sin cambiarlos",
    "el runtime controla que puede hacer el agente",
    "cada tool call es validada por el runtime",
    "la memoria del agente esta en el prompt no en los pesos",
    "backpropagation ajusta los pesos para reducir el error",
]

# Construir vocabulario
todas_palabras = set()
for t in textos:
    todas_palabras.update(t.split())

palabra_a_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
for i, palabra in enumerate(sorted(todas_palabras), start=3):
    palabra_a_id[palabra] = i

id_a_palabra = {v: k for k, v in palabra_a_id.items()}
vocab_size = len(palabra_a_id)

print(f"Vocabulario: {vocab_size} tokens")
print(f"Ejemplo: 'el' -> {palabra_a_id.get('el', '?')}")

# Tokenizar textos
def tokenizar_texto(texto):
    return [palabra_a_id["<bos>"]] + [palabra_a_id[p] for p in texto.split()] + [palabra_a_id["<eos>"]]

secuencias = [tokenizar_texto(t) for t in textos]

# Padding para que todas las secuencias tengan la misma longitud
max_len = max(len(s) for s in secuencias)
for s in secuencias:
    while len(s) < max_len:
        s.append(palabra_a_id["<pad>"])

datos = torch.tensor(secuencias)

# Hiperparametros del modelo
D_MODEL = 64       # Dimension de embeddings (GPT-4: ~12,000)
NUM_HEADS = 4      # Cabezas de atencion (GPT-4: ~128)
NUM_LAYERS = 3     # Bloques de transformer (GPT-4: ~120)
D_FF = 128         # Dimension del feed-forward (GPT-4: ~48,000)
MAX_SEQ_LEN = max_len

# Crear modelo
modelo = MiniTransformer(vocab_size, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LEN)
num_params = sum(p.numel() for p in modelo.parameters())
print(f"Parametros totales: {num_params:,}")
print(f"(GPT-4 tiene ~1,800,000,000,000 parametros)")
print()

# Optimizer
optimizer = torch.optim.AdamW(modelo.parameters(), lr=3e-3)

# Entrenamiento
print("Entrenando...")
EPOCAS = 500

for epoca in range(EPOCAS):
    modelo.train()

    # Input: todos los tokens excepto el ultimo
    # Target: todos los tokens excepto el primero (shifted right)
    input_ids = datos[:, :-1]
    target_ids = datos[:, 1:]

    logits = modelo(input_ids)

    # Cross-entropy loss (ignorando padding)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=palabra_a_id["<pad>"],
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoca + 1) % 100 == 0:
        print(f"  Epoca {epoca + 1:>4}: loss = {loss.item():.4f}")

# ---- Generar texto ----
print()
print("=" * 50)
print("GENERACION DE TEXTO")
print("=" * 50)

# Generar a partir de "<bos> el"
inicio = torch.tensor([[palabra_a_id["<bos>"], palabra_a_id["el"]]])
generado = modelo.generar(inicio, max_nuevos_tokens=8, temperatura=0.5)

tokens_generados = generado[0].tolist()
texto_generado = " ".join([id_a_palabra.get(t, "?") for t in tokens_generados])
print(f"Generado: {texto_generado}")

# Varias generaciones con diferentes inicios
for inicio_texto in ["la", "un", "los"]:
    if inicio_texto in palabra_a_id:
        inicio = torch.tensor([[palabra_a_id["<bos>"], palabra_a_id[inicio_texto]]])
        generado = modelo.generar(inicio, max_nuevos_tokens=8, temperatura=0.7)
        tokens_generados = generado[0].tolist()
        texto_generado = " ".join([id_a_palabra.get(t, "?") for t in tokens_generados])
        print(f"Generado: {texto_generado}")

print()
print("NOTA: Este mini-transformer tiene la MISMA arquitectura que GPT.")
print(f"La diferencia es la escala: {num_params:,} parametros vs 1,800,000,000,000.")
print("El principio es identico: predecir el siguiente token.")
```


## 2.10 Que acabamos de construir

Lo que acabas de ver es un GPT en miniatura. No es una metafora:

| Componente | Mini-Transformer | GPT-4 |
|---|---|---|
| Embeddings | 64 dimensiones | ~12,000 dimensiones |
| Heads | 4 | ~128 |
| Layers | 3 | ~120 |
| Feed-forward | 128 dim | ~48,000 dim |
| Parametros | ~100K | ~1.8T |
| Dataset | 10 frases | Trillones de tokens |
| Hardware | CPU, segundos | Miles de GPUs, meses |

La arquitectura es la misma. La diferencia es escala y datos.

---

*En la siguiente seccion: como un transformer entrenado se convierte
en un LLM que puede mantener conversaciones.*
