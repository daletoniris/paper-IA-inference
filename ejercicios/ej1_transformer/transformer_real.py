#!/usr/bin/env python3
"""
Ejercicio 1: Transformer Decoder-Only REAL desde cero.
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

A diferencia del mini_transformer.py del paper (juguete de 110K params
con 15 frases), este es un Transformer REAL con:

  - Tokenizador BPE construido desde cero (como GPT-2/GPT-4)
  - RMSNorm en vez de LayerNorm (como LLaMA, Qwen, Mistral)
  - RoPE: Rotary Position Embeddings (como LLaMA, Qwen, Mistral)
  - SwiGLU: SiLU + Gated Linear Unit (como LLaMA)
  - KV-Cache para generacion eficiente
  - Entrenado en texto REAL (no frases inventadas)

Entrenable en una Raspberry Pi 5 (CPU, 8GB RAM) en ~5 minutos.
~2.5M parametros -- 20x mas grande que el del paper, pero la
arquitectura es IDENTICA a la de LLaMA/Qwen/Mistral.

Autor: TokioAI (ejercicio practico del paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
from collections import Counter


# =====================================================================
# PARTE 1: TOKENIZADOR BPE DESDE CERO
# =====================================================================
# Los LLMs reales usan BPE (Byte Pair Encoding). GPT-2 lo popularizo.
# La idea: empezas con caracteres individuales y vas fusionando los
# pares mas frecuentes hasta tener un vocabulario del tamaño que quieras.
#
# Ejemplo: "hola hola mundo" con caracteres ['h','o','l','a',' ','m','u','n','d']
# Paso 1: par mas frecuente = ('h','o') -> fusionar en 'ho'
# Paso 2: par mas frecuente = ('ho','l') -> fusionar en 'hol'
# Paso 3: par mas frecuente = ('hol','a') -> fusionar en 'hola'
# ... y asi hasta llegar al tamaño de vocab deseado.
# =====================================================================

class BPETokenizer:
    """
    Tokenizador BPE (Byte Pair Encoding) construido desde cero.
    
    Asi funciona el tokenizador de GPT-2, GPT-3, GPT-4, LLaMA, etc.
    La unica diferencia es la escala: GPT-4 tiene ~100K tokens,
    nosotros usamos ~1000 para que sea entrenable en la Pi.
    """

    def __init__(self):
        self.merges = {}          # (token_a, token_b) -> nuevo_token
        self.vocab = {}           # id -> bytes
        self.inverse_vocab = {}   # bytes -> id
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

    def train(self, text, vocab_size=1000, verbose=True):
        """
        Entrena el tokenizador BPE en un texto.
        
        Proceso:
        1. Convertir texto a bytes (UTF-8)
        2. Contar frecuencia de cada par de bytes adyacentes
        3. Fusionar el par mas frecuente -> nuevo token
        4. Repetir hasta llegar a vocab_size
        
        NOTA: Para eficiencia en CPU, usamos solo los primeros 100K chars
        para entrenar el tokenizador. Es suficiente para aprender los
        patrones del idioma. GPT-4 entrena su BPE en terabytes.
        """
        # Usar subset para entrenar BPE (el vocab se satura rapido)
        # 30K chars es suficiente -- el BPE aprende los patrones comunes rapido
        train_text = text[:30_000] if len(text) > 30_000 else text

        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRENANDO TOKENIZADOR BPE")
            print(f"{'='*60}")
            print(f"Texto total: {len(text):,} caracteres")
            print(f"Texto para BPE: {len(train_text):,} caracteres")
            print(f"Vocab objetivo: {vocab_size} tokens")

        # Paso 1: Empezar con bytes individuales (0-255)
        # Cada byte es un token inicial. UTF-8 usa bytes 0-255.
        tokens = list(train_text.encode("utf-8"))

        # Vocabulario base: 4 tokens especiales + 256 bytes
        num_special = len(self.special_tokens)
        self.vocab = {i: bytes([0]) for i in range(num_special)}  # especiales
        for b in range(256):
            self.vocab[b + num_special] = bytes([b])
        
        # Convertir tokens a IDs (byte + offset por tokens especiales)
        token_ids = [t + num_special for t in tokens]

        num_merges = vocab_size - 256 - num_special
        if verbose:
            print(f"Tokens iniciales: {len(token_ids):,}")
            print(f"Merges a hacer: {num_merges}")
            print()

        for i in range(num_merges):
            # Contar pares de forma eficiente
            pairs = {}
            prev = token_ids[0]
            for j in range(1, len(token_ids)):
                curr = token_ids[j]
                pair = (prev, curr)
                pairs[pair] = pairs.get(pair, 0) + 1
                prev = curr

            if not pairs:
                break

            # Par mas frecuente
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            if best_count < 2:
                break

            # Crear nuevo token
            new_id = 256 + num_special + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Aplicar merge (optimizado)
            new_tokens = []
            j = 0
            while j < len(token_ids):
                if j < len(token_ids) - 1 and token_ids[j] == best_pair[0] and token_ids[j + 1] == best_pair[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(token_ids[j])
                    j += 1
            token_ids = new_tokens

            if verbose and (i + 1) % 100 == 0:
                decoded = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  Merge {i+1:>4}: {repr(decoded):>20} (freq={best_count:>5}, tokens={len(token_ids):,})")

        # Construir vocabulario inverso
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"\nVocabulario final: {len(self.vocab)} tokens")
            ratio = len(text.encode("utf-8")) / len(token_ids)
            print(f"Ratio de compresion: {ratio:.2f}x (cada token = ~{ratio:.1f} bytes)")

        return self

    def encode(self, text):
        """Tokeniza texto usando los merges aprendidos."""
        token_ids = [b + len(self.special_tokens) for b in text.encode("utf-8")]

        # Aplicar merges en orden
        for pair, new_id in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            token_ids = new_tokens

        return token_ids

    def decode(self, ids):
        """Convierte IDs de vuelta a texto."""
        byte_list = b""
        for id in ids:
            if id in self.special_tokens.values():
                continue
            if id in self.vocab:
                byte_list += self.vocab[id]
        return byte_list.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        """Guarda el tokenizador."""
        data = {
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Carga el tokenizador."""
        with open(path) as f:
            data = json.load(f)
        self.merges = {tuple(int(x) for x in k.split(",")): v for k, v in data["merges"].items()}
        self.special_tokens = data["special_tokens"]
        # Reconstruir vocab
        num_special = len(self.special_tokens)
        self.vocab = {i: bytes([0]) for i in range(num_special)}
        for b in range(256):
            self.vocab[b + num_special] = bytes([b])
        for pair, new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        return self


# =====================================================================
# PARTE 2: COMPONENTES DEL TRANSFORMER MODERNO
# =====================================================================
# Estos son los bloques que usan LLaMA, Qwen, Mistral, etc.
# La diferencia con el Transformer original (2017):
#   - RMSNorm en vez de LayerNorm (mas rapido, funciona igual)
#   - RoPE en vez de positional embeddings (mejor para secuencias largas)
#   - SwiGLU en vez de FFN clasico (mejor rendimiento)
#   - Pre-norm en vez de post-norm (mas estable en training)
# =====================================================================


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Usado por LLaMA, Qwen, Mistral en vez de LayerNorm.
    Es mas simple y rapido: solo normaliza por la raiz de la media
    de los cuadrados, sin restar la media (no tiene bias).
    
    LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:   x / sqrt(mean(x^2) + eps) * gamma
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """
    Precomputa las frecuencias para RoPE (Rotary Position Embeddings).
    
    RoPE codifica la posicion ROTANDO los vectores de query y key.
    La idea genial: si rotas q y k por un angulo que depende de su
    posicion, el producto punto q . k depende de la DIFERENCIA de
    posiciones, no de las posiciones absolutas.
    
    Esto es mejor que positional embeddings porque:
    1. Generaliza a secuencias mas largas que las vistas en training
    2. La atencion relativa emerge naturalmente
    3. No agrega parametros
    
    Usado por: LLaMA, Qwen, Mistral, Gemma, etc.
    """
    # Frecuencias: theta^(-2i/d) para i = 0, 1, ..., d/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Posiciones: 0, 1, 2, ..., max_seq_len - 1
    t = torch.arange(max_seq_len).float()
    # Producto externo: angulos para cada (posicion, dimension)
    angles = torch.outer(t, freqs)
    # Forma compleja: cos + i*sin
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs):
    """
    Aplica RoPE a un tensor x.
    
    Convierte pares de dimensiones (x0, x1) a numeros complejos,
    los multiplica por e^(i*theta) (rotacion), y vuelve a reales.
    """
    # x: (B, num_heads, T, d_head)
    B, H, T, D = x.shape
    
    # Tomar solo las frecuencias para las posiciones que necesitamos
    freqs = freqs[:T].to(x.device)
    
    # Convertir a complejo: pares de dims -> numeros complejos
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    
    # Rotar multiplicando por e^(i*theta)
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    x_rotated = x_complex * freqs
    
    # Volver a reales
    return torch.view_as_real(x_rotated).reshape(B, H, T, D).type_as(x)


class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention con RoPE y KV-Cache.
    
    Identica a la atencion de LLaMA/Qwen/Mistral.
    
    KV-Cache: durante generacion, no necesitas recalcular K y V para
    los tokens anteriores. Los guardas en cache y solo calculas para
    el token nuevo. Esto hace la generacion O(n) en vez de O(n^2).
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, rope_freqs, kv_cache=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Aplicar RoPE a Q y K (no a V -- la posicion solo importa para atencion)
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # KV-Cache: concatenar con cache anterior si existe
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        # Atencion: Q @ K^T / sqrt(d) con mascara causal
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Mascara causal: cada token solo puede ver tokens anteriores
        T_q = q.size(2)
        T_k = k.size(2)
        mask = torch.triu(torch.ones(T_q, T_k, device=x.device), diagonal=T_k - T_q + 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out), new_cache


class SwiGLU(nn.Module):
    """
    SwiGLU: SiLU-Gated Linear Unit.
    
    Usado por LLaMA, Qwen, Mistral en vez del FFN clasico.
    Es como un FFN pero con una "compuerta" que controla el flujo
    de informacion:
    
    FFN clasico: Linear -> ReLU -> Linear
    SwiGLU:      gate = SiLU(Linear_gate(x))
                 value = Linear_up(x)
                 output = Linear_down(gate * value)
    
    La compuerta permite al modelo "elegir" que informacion pasa.
    Funciona mejor que FFN clasico en la practica.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU = x * sigmoid(x)
        up = self.w_up(x)
        return self.w_down(gate * up)


class TransformerBlock(nn.Module):
    """
    Bloque Transformer con Pre-Norm (como LLaMA/Qwen/Mistral).
    
    Pre-norm vs Post-norm:
    - Original (post-norm): x = LayerNorm(x + Attn(x))
    - Moderno (pre-norm):   x = x + Attn(RMSNorm(x))
    
    Pre-norm es mas estable para entrenar modelos grandes.
    Todos los LLMs modernos lo usan.
    """

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = SelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)

    def forward(self, x, rope_freqs, kv_cache=None):
        # Pre-norm + Residual
        h, new_cache = self.attn(self.norm_attn(x), rope_freqs, kv_cache)
        x = x + h
        x = x + self.ffn(self.norm_ffn(x))
        return x, new_cache


# =====================================================================
# PARTE 3: EL MODELO COMPLETO
# =====================================================================

class TransformerLM(nn.Module):
    """
    Transformer Decoder-Only para modelado de lenguaje.
    
    Arquitectura IDENTICA a LLaMA / Qwen / Mistral.
    La unica diferencia es la escala:
    
    | Componente   | Este modelo | LLaMA-7B    | Qwen-72B     |
    |-------------|-------------|-------------|--------------|
    | d_model     | 128         | 4,096       | 8,192        |
    | num_heads   | 4           | 32          | 64           |
    | num_layers  | 4           | 32          | 80           |
    | d_ff        | 384         | 11,008      | 24,576       |
    | vocab_size  | ~1,000      | 32,000      | 152,064      |
    | params      | ~2.5M       | 6,738M      | 72,710M      |
    | RAM train   | ~200MB      | ~52GB       | ~580GB       |
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4,
                 num_layers=4, d_ff=384, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Embedding de tokens (sin positional embedding -- RoPE lo reemplaza)
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Bloques Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # Norma final + cabeza de salida
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding y cabeza comparten pesos
        # Esto reduce parametros y mejora el rendimiento
        self.head.weight = self.tok_emb.weight

        # Precomputar frecuencias RoPE
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(d_model // num_heads, max_seq_len)
        )

        # Inicializacion de pesos (importante para convergencia)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_caches=None):
        """
        Forward pass.
        
        idx: (B, T) tensor de token IDs
        kv_caches: lista de KV-caches por capa (para generacion)
        
        Retorna: logits (B, T, vocab_size), new_kv_caches
        """
        B, T = idx.shape
        x = self.tok_emb(idx)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, self.rope_freqs, cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """
        Genera texto token por token con KV-Cache.
        
        Sin KV-Cache: cada token nuevo requiere re-procesar TODA la secuencia.
        Con KV-Cache: solo procesamos el ultimo token y reutilizamos los K,V
        de los tokens anteriores. Speedup lineal.
        
        idx: (1, T) tensor con tokens de contexto
        """
        self.eval()
        kv_caches = None

        for i in range(max_new_tokens):
            # Primera iteracion: procesar todo el contexto
            # Siguientes: solo el ultimo token (KV-cache tiene el resto)
            if kv_caches is None:
                input_ids = idx
            else:
                input_ids = idx[:, -1:]

            logits, kv_caches = self(input_ids, kv_caches)
            logits = logits[:, -1, :] / temperature

            # Top-K sampling: solo considerar los K tokens mas probables
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=-1)

            # Parar si generamos <eos>
            if next_token.item() == 2:  # <eos>
                break

        return idx

    def count_params(self):
        """Cuenta parametros totales y por componente."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()
        attention = sum(p.numel() for b in self.blocks for p in b.attn.parameters())
        ffn = sum(p.numel() for b in self.blocks for p in b.ffn.parameters())
        norms = sum(p.numel() for b in self.blocks for p in b.norm_attn.parameters()) + \
                sum(p.numel() for b in self.blocks for p in b.norm_ffn.parameters()) + \
                sum(p.numel() for p in self.norm.parameters())
        return {
            "total": total,
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "norms": norms,
            # head no cuenta por weight tying
        }


# =====================================================================
# PARTE 4: DATASET Y ENTRENAMIENTO
# =====================================================================

def download_dataset():
    """
    Descarga TinyShakespeare: 1MB de texto de Shakespeare.
    Es el dataset clasico para probar language models.
    """
    path = os.path.join(os.path.dirname(__file__), "data_tinystory.txt")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()

    print("Descargando dataset...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error descargando: {e}")
        print("Usando dataset local de fallback...")
        # Fallback: generar texto suficiente para entrenar
        fallback = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answered it.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that loured upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
""" * 20  # Repetir para tener suficiente texto
        with open(path, "w") as f:
            f.write(fallback)
        return fallback


class TextDataset:
    """Dataset para language modeling."""

    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloader(dataset, batch_size, shuffle=True):
    """DataLoader manual (sin dependencias extra)."""
    indices = list(range(len(dataset)))
    if shuffle:
        import random
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        xs, ys = [], []
        for idx in batch_idx:
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


# =====================================================================
# PARTE 5: TRAINING LOOP
# =====================================================================

def train(model, tokenizer, text, epochs=3, batch_size=16, seq_len=128,
          lr=3e-4, log_interval=50):
    """
    Entrena el modelo en el texto dado.
    
    Proceso (identico a como se entrenan los LLMs reales):
    1. Tokenizar todo el texto
    2. Crear secuencias de largo fijo
    3. Para cada batch:
       a. Forward pass: predecir siguiente token
       b. Calcular loss (cross-entropy)
       c. Backward pass: calcular gradientes
       d. Actualizar pesos con AdamW
    """
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO")
    print(f"{'='*60}")

    # Tokenizar
    tokens = tokenizer.encode(text)
    n_tokens = len(tokens)
    
    # Split train/val (90/10)
    split = int(0.9 * n_tokens)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    params = model.count_params()
    print(f"Tokens de entrenamiento: {len(train_tokens):,}")
    print(f"Tokens de validacion:    {len(val_tokens):,}")
    print(f"Secuencias de training:  {len(train_ds):,}")
    print(f"Batch size:              {batch_size}")
    print(f"Largo de secuencia:      {seq_len}")
    print(f"Learning rate:           {lr}")
    print(f"Epochs:                  {epochs}")
    print(f"Parametros del modelo:   {params['total']:,}")
    print(f"  - Embedding:           {params['embedding']:,}")
    print(f"  - Attention:           {params['attention']:,}")
    print(f"  - FFN (SwiGLU):        {params['ffn']:,}")
    print(f"  - Norms (RMSNorm):     {params['norms']:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Cosine annealing scheduler
    total_steps = epochs * (len(train_ds) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=lr/10)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in create_dataloader(train_ds, batch_size):
            logits, _ = model(batch_x)
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                batch_y.reshape(-1),
                ignore_index=0  # ignorar <pad>
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (previene explosion de gradientes)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % log_interval == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{epochs} | Batch {n_batches:>4} | "
                      f"Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Tiempo: {elapsed:.0f}s")

        train_loss = total_loss / max(n_batches, 1)

        # --- Validacion ---
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in create_dataloader(val_ds, batch_size, shuffle=False):
                logits, _ = model(batch_x)
                loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size),
                    batch_y.reshape(-1),
                    ignore_index=0
                )
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - start_time

        # Perplexity: exp(loss). Mide que tan "confundido" esta el modelo.
        # Perplexity = vocab_size significa que adivina al azar.
        # Perplexity < 100 ya es bastante bueno para nuestro modelo.
        ppl = math.exp(min(val_loss, 20))  # cap para evitar overflow

        print(f"\n  >>> Epoch {epoch+1} completada: "
              f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"Perplexity={ppl:.1f} | Tiempo={elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Guardar mejor modelo
            save_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Mejor modelo guardado (val_loss={val_loss:.4f})")

        print()

    elapsed = time.time() - start_time
    print(f"Entrenamiento completado en {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Mejor val loss: {best_val_loss:.4f} (perplexity: {math.exp(min(best_val_loss, 20)):.1f})")

    return best_val_loss


# =====================================================================
# PARTE 6: MAIN -- EJECUTAR TODO
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EJERCICIO 1: TRANSFORMER REAL DESDE CERO")
    print("Paper: Hackers Artificiales o Sistemas Mal Disenados?")
    print("=" * 60)
    print()
    print("Este ejercicio construye un Transformer REAL con la")
    print("misma arquitectura que LLaMA, Qwen y Mistral.")
    print("La unica diferencia es la escala.")
    print()

    # 1. Descargar/cargar dataset
    text = download_dataset()
    print(f"Dataset: {len(text):,} caracteres")
    print(f"Preview: {text[:200]}...")
    print()

    # Usar los primeros 50K chars para training
    # Suficiente para aprender patrones de Shakespeare en la Pi.
    # En un GPU se usarian los 1.1M completos.
    text = text[:50_000]
    print(f"Usando: {len(text):,} caracteres para training")

    # 2. Entrenar tokenizador BPE
    VOCAB_SIZE = 400  # 400 tokens BPE -- rapido de entrenar en Pi
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=VOCAB_SIZE)

    # Demo del tokenizador
    print(f"\n--- Demo del tokenizador ---")
    sample = "To be, or not to be, that is the question"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"  Texto:     {sample}")
    print(f"  Tokens:    {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
    print(f"  N tokens:  {len(encoded)}")
    print(f"  Decoded:   {decoded}")

    # Guardar tokenizador
    tok_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"  Guardado en: {tok_path}")

    # 3. Crear modelo
    print(f"\n{'='*60}")
    print(f"CREANDO MODELO")
    print(f"{'='*60}")

    D_MODEL = 96
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 256
    MAX_SEQ = 64  # Secuencias mas cortas = training mas rapido

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )

    params = model.count_params()
    print(f"\nArquitectura:")
    print(f"  d_model:     {D_MODEL}")
    print(f"  num_heads:   {NUM_HEADS}")
    print(f"  d_head:      {D_MODEL // NUM_HEADS}")
    print(f"  num_layers:  {NUM_LAYERS}")
    print(f"  d_ff:        {D_FF}")
    print(f"  vocab_size:  {tokenizer.vocab_size}")
    print(f"  max_seq_len: {MAX_SEQ}")
    print(f"\nParametros:")
    print(f"  Total:       {params['total']:>10,}")
    print(f"  Embedding:   {params['embedding']:>10,}")
    print(f"  Attention:   {params['attention']:>10,}")
    print(f"  FFN:         {params['ffn']:>10,}")
    print(f"  Norms:       {params['norms']:>10,}")
    print(f"\nComparacion:")
    print(f"  Este modelo:  {params['total']:>14,} params")
    print(f"  GPT-2:        124,000,000 params ({124_000_000//params['total']}x mas grande)")
    print(f"  LLaMA-7B:   6,738,000,000 params ({6_738_000_000//params['total']}x mas grande)")
    print(f"  GPT-4:    1,800,000,000,000 params (~{1_800_000_000_000//params['total']}x mas grande)")

    # 4. Entrenar
    # 2 epochs es suficiente para ver convergencia en la Pi
    best_loss = train(
        model, tokenizer, text,
        epochs=2,
        batch_size=32,
        seq_len=MAX_SEQ,
        lr=1e-3,
        log_interval=50,
    )

    # 5. Generar texto
    print(f"\n{'='*60}")
    print(f"GENERACION DE TEXTO")
    print(f"{'='*60}")
    print(f"Generando con KV-Cache (como los LLMs reales)...")
    print()

    # Cargar mejor modelo
    best_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    prompts = [
        "KING HENRY",
        "To be or not",
        "My lord",
        "The king",
        "What is",
    ]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])

        t0 = time.time()
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
        t1 = time.time()

        generated = tokenizer.decode(output_ids[0].tolist())
        n_new = output_ids.shape[1] - input_ids.shape[1]
        tok_s = n_new / (t1 - t0)

        print(f"  Prompt: \"{prompt}\"")
        print(f"  Output: \"{generated[:200]}\"")
        print(f"  ({n_new} tokens en {t1-t0:.2f}s = {tok_s:.0f} tok/s)")
        print()

    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"""
Lo que acabas de construir y ejecutar:

1. TOKENIZADOR BPE desde cero
   - El mismo algoritmo que usan GPT-2/GPT-4/LLaMA
   - Aprende subpalabras automaticamente del texto
   - {tokenizer.vocab_size} tokens (GPT-4 usa ~100K)

2. TRANSFORMER con arquitectura moderna:
   - RMSNorm (como LLaMA/Qwen/Mistral)
   - RoPE (Rotary Position Embeddings)
   - SwiGLU (Gated FFN)
   - KV-Cache para generacion eficiente
   - Pre-norm con conexiones residuales

3. ENTRENAMIENTO real con AdamW + cosine annealing
   - Dataset real (Shakespeare, {len(text):,} chars)
   - Train/val split
   - Gradient clipping
   - {params['total']:,} parametros

4. GENERACION token por token con sampling
   - Top-K sampling
   - Temperature control
   - KV-Cache (O(n) en vez de O(n^2))

La arquitectura es IDENTICA a LLaMA/Qwen/Mistral.
La diferencia es SOLO escala: mas parametros + mas datos.
No hay magia. No hay "inteligencia". Es matematica.
""")
