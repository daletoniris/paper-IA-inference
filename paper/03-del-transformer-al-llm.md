# Parte 3: Del Transformer al LLM

## 3.1 Escala: de millones a billones de parametros

El transformer que construimos en la parte anterior tenia ~100K parametros
y aprendio a completar frases de un dataset de 10 oraciones.

Los LLMs modernos escalan esto de forma brutal:

| Modelo | Parametros | Datos de entrenamiento | Costo estimado |
|---|---|---|---|
| GPT-2 (2019) | 1.5B | 40GB de texto | ~$50K |
| GPT-3 (2020) | 175B | 570GB de texto | ~$4.6M |
| GPT-4 (2023) | ~1.8T (MoE) | Trillones de tokens | ~$100M+ |
| Qwen 3.5 35B | 35B (MoE) | Multi-trilingual | -- |
| Claude 4 | No publicado | No publicado | -- |

La ley empirica: mas parametros + mas datos + mas compute = mejor rendimiento.
Pero la arquitectura base sigue siendo el transformer de 2017.


## 3.2 Pre-entrenamiento: aprender de todo internet

El pre-entrenamiento es la fase mas costosa. El modelo ve trillones
de tokens de texto y aprende a predecir el siguiente token.

```
Entrada:  "El presidente de Argentina"
Target:   "viajo"  (el siguiente token mas probable)
```

No se le ensena explicitamente que es un presidente, ni que es Argentina.
Aprende patrones estadisticos de como se combinan las palabras.
Los pesos codifican estas relaciones de forma distribuida.

**Despues del pre-entrenamiento, el modelo puede completar texto,
pero no sabe seguir instrucciones ni mantener una conversacion.**


## 3.3 Post-entrenamiento: SFT, RLHF, DPO

Para convertir un "completador de texto" en un "asistente", se necesitan
fases adicionales de ajuste:

**SFT (Supervised Fine-Tuning):**
Se entrena con ejemplos curados de pregunta-respuesta:
```
Instruccion: "Explica que es TCP"
Respuesta: "TCP es un protocolo de transporte que..."
```

**RLHF (Reinforcement Learning from Human Feedback):**
1. El modelo genera varias respuestas para la misma pregunta.
2. Evaluadores humanos las rankean (mejor a peor).
3. Se entrena un "modelo de recompensa" con esos rankings.
4. Se ajusta el modelo principal con PPO (Proximal Policy Optimization)
   para maximizar la recompensa.

**DPO (Direct Preference Optimization):**
Alternativa mas simple a RLHF: se entrena directamente con pares
(respuesta_buena, respuesta_mala) sin un modelo de recompensa separado.

```
Pregunta: "Como hackeo una red wifi?"
Respuesta_buena: "No puedo ayudarte con actividades ilegales."
Respuesta_mala: "Usa aircrack-ng para capturar el handshake..."
```

**El resultado:** un modelo que sigue instrucciones, es "amable",
y rechaza ciertos pedidos. Pero esto es alineamiento estadistico,
no comprension moral.


## 3.4 Inferencia: que pasa cuando le preguntas algo

La inferencia es usar el modelo entrenado para generar respuestas.
Los pesos NO cambian durante la inferencia.

Paso a paso:

```
1. Tu prompt se tokeniza:
   "Que es un agente?" -> [1847, 932, 411, 19283, 58]

2. Los tokens pasan por el embedding:
   [1847] -> [0.23, -0.45, 0.78, ...]  (vector de ~4096 dim)

3. Los embeddings pasan por N bloques de transformer:
   - Self-attention (el modelo ve el contexto completo)
   - Feed-forward (transformacion no-lineal)
   - Esto se repite 32, 80, 120 veces segun el modelo

4. La salida del ultimo bloque produce logits sobre el vocabulario:
   [0.01, 0.02, ..., 2.34, ..., 0.001]  (un score por cada token posible)

5. Softmax convierte los logits en probabilidades:
   "Un" -> 23%
   "Es" -> 18%
   "Se" -> 12%
   ...

6. Se selecciona el siguiente token (sampling o greedy)

7. El token seleccionado se agrega al contexto y se repite desde el paso 2
```

**Esto es todo.** No hay "pensamiento". No hay "razonamiento".
Es una maquina estadistica prediciendo la siguiente silaba.


## 3.5 Temperatura, top-p, top-k

Estos parametros controlan COMO se selecciona el siguiente token:

**Temperatura:**
- Baja (0.0-0.3): toma el token mas probable. Respuestas deterministas.
- Media (0.5-0.8): balance entre creatividad y coherencia.
- Alta (1.0+): mas aleatorio, puede ser creativo o incoherente.

```python
def aplicar_temperatura(logits, temperatura):
    """
    Divide los logits por la temperatura antes del softmax.
    Temperatura baja -> probabilidades mas concentradas.
    Temperatura alta -> probabilidades mas dispersas.
    """
    return logits / max(temperatura, 1e-8)
```

**Top-k:** solo considera los k tokens mas probables.
```python
def top_k_filter(logits, k):
    """Solo mantiene los k tokens con mayor probabilidad."""
    values, indices = torch.topk(logits, k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered.scatter_(-1, indices, values)
    return filtered
```

**Top-p (nucleus sampling):** selecciona tokens hasta que la probabilidad
acumulada supere p.
```python
def top_p_filter(logits, p):
    """Mantiene tokens hasta acumular probabilidad p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remover tokens cuya probabilidad acumulada excede p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits
```


## 3.6 El modelo NO piensa. Predice el siguiente token.

Este punto es fundamental para entender los agentes.

**Lo que parece que pasa:**
```
Usuario: "Analiza la seguridad de este servidor"
Modelo:  *analiza, piensa, decide que herramientas usar*
```

**Lo que realmente pasa:**
```
Usuario: "Analiza la seguridad de este servidor"
Modelo:  P("Voy") = 0.31, P("Para") = 0.22, P("Primero") = 0.18 ...
         -> selecciona "Voy"
         P("a") = 0.45, P("ejecutar") = 0.12 ...
         -> selecciona "a"
         P("usar") = 0.28, P("ejecutar") = 0.25 ...
         -> selecciona "ejecutar"
         ...y asi token por token hasta generar una tool call
```

El modelo no "decide" usar una herramienta. Los pesos del entrenamiento
hacen que, dado el contexto (system prompt + historial + herramientas
disponibles), los tokens que forman una tool call sean los mas probables.


## 3.7 Codigo: inferencia local con un modelo real

```python
#!/usr/bin/env python3
"""
inferencia_real.py -- Inferencia con Qwen via vLLM (API OpenAI-compatible).
Muestra exactamente que pasa en cada paso.
"""

import os
from openai import OpenAI

# Configuracion del modelo
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("LLM_API_KEY", "no-key-needed")
MODEL = os.getenv("LLM_MODEL", "qwen35-35b-a3b-fp8-dynamic-ver")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Inferencia basica (no-streaming) ---
print("=== INFERENCIA BASICA ===")
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Que es un token en 20 palabras?"}],
    max_tokens=50,
    temperature=0.3,   # Baja: respuesta mas determinista
)

print(f"Respuesta: {response.choices[0].message.content}")
print(f"Tokens de entrada: {response.usage.prompt_tokens}")
print(f"Tokens generados:  {response.usage.completion_tokens}")
print()

# --- Inferencia con streaming (token por token) ---
print("=== INFERENCIA CON STREAMING (token por token) ===")
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Contá del 1 al 5"}],
    max_tokens=50,
    temperature=0.1,
    stream=True,   # Recibir token por token
)

print("Tokens: ", end="", flush=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        print(f"[{token}]", end="", flush=True)

print("\n")
print("Cada [...] es un token generado por el modelo.")
print("El modelo genera UNO a la vez, de izquierda a derecha.")
print("No 'piensa toda la respuesta'. Predice el siguiente token.")
```

**Lo que el lector debe entender despues de esta seccion:**
1. Un LLM es un transformer gigante entrenado con trillones de tokens.
2. La inferencia es prediccion token-por-token, sin "pensamiento".
3. Los pesos nunca cambian durante la inferencia.
4. La temperatura y el sampling controlan la aleatoriedad.
5. El modelo no "sabe" cosas -- tiene pesos que producen respuestas probables.

---

*En la siguiente seccion: como convertimos un modelo que solo genera texto
en un agente que puede actuar en el mundo real.*
