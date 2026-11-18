#!/usr/bin/env python3
"""
Ejercicio 4: Inferencia Real -- KV-Cache, Sampling, y Probabilidades
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

Este ejercicio expone las TRIPAS de la inferencia. Cuando le preguntas
algo a ChatGPT/Claude, internamente pasa exactamente esto:

  1. Tokenizacion: tu texto se convierte en numeros
  2. Forward pass: los numeros pasan por N capas de transformer
  3. Logits: la ultima capa produce un score para cada token del vocab
  4. Sampling: se elige el siguiente token segun temperatura/top-k/top-p
  5. KV-Cache: se reutilizan calculos previos (no re-procesar todo)
  6. Repeat: el token generado se agrega al contexto y se repite

Aqui lo hacemos VISIBLE:
  - Mostramos las probabilidades de cada token candidato
  - Comparamos greedy vs random vs top-k vs top-p vs nucleus
  - Implementamos KV-cache y medimos el speedup real
  - Visualizamos como la temperatura cambia la distribucion

Tiempo estimado: ~5-10 min en Pi 5 (CPU).

Autor: TokioAI (ejercicio practico del paper)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

# Reutilizamos la infraestructura de Ej1/Ej2/Ej3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej1_transformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej2_training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej3_mini_llm'))
from transformer_real import BPETokenizer, precompute_rope_freqs
from train_real import TransformerLMv2
from mini_llm import ChatTokenizer


# =====================================================================
# PARTE 1: ANATOMIA DE UN FORWARD PASS
# =====================================================================
# Cuando el modelo genera un token, internamente pasa:
#   1. Embedding lookup: token_id -> vector de 128 dimensiones
#   2. 4 bloques de Transformer:
#      a. RMSNorm (normalizar)
#      b. Self-Attention (el token "mira" todos los anteriores)
#      c. RMSNorm (normalizar otra vez)
#      d. SwiGLU FFN (transformacion no-lineal)
#   3. RMSNorm final
#   4. Linear head: vector 128-dim -> logits de 1005 dimensiones
#      (un score para cada token posible del vocabulario)
#   5. Softmax: logits -> probabilidades (suman 1.0)
# =====================================================================


def show_forward_pass(model, tokenizer, text, device='cpu'):
    """
    Muestra EXACTAMENTE que pasa en cada capa del modelo.
    """
    model.eval()
    tokens = tokenizer.base_tokenizer.encode(text)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n  Texto de entrada: \"{text}\"")
    print(f"  Tokens: {tokens}")
    print(f"  Token count: {len(tokens)}")
    print()

    with torch.no_grad():
        # Paso 1: Embedding
        h = model.tok_emb(ids)
        h = model.emb_dropout(h)
        print(f"  [Embedding] Input: {ids.shape} -> Output: {h.shape}")
        print(f"    Cada token se convierte en un vector de {h.shape[-1]} dimensiones")
        print(f"    Norma promedio del embedding: {h.norm(dim=-1).mean().item():.4f}")

        # Paso 2: Bloques de Transformer
        for i, block in enumerate(model.blocks):
            h_prev = h.clone()

            # Attention
            h_normed = block.norm_attn(h)
            attn_out, _ = block.attn(h_normed, model.rope_freqs, None)
            h = h + block.res_dropout(attn_out)

            # FFN
            h_normed = block.norm_ffn(h)
            ffn_out = block.ffn(h_normed)
            h = h + block.res_dropout(ffn_out)

            # Delta: cuanto cambio
            delta = (h - h_prev).norm().item()
            print(f"  [Bloque {i}] Norma del cambio: {delta:.4f}")

        # Paso 3: Norm final + head
        h = model.norm(h)
        logits = model.head(h)  # [1, T, V]

        print(f"\n  [Output Head] {h.shape} -> {logits.shape}")
        print(f"    {logits.shape[-1]} logits = un score por cada token del vocabulario")

        # Paso 4: Logits del ULTIMO token (prediccion del siguiente)
        last_logits = logits[0, -1, :]  # [V]
        probs = F.softmax(last_logits, dim=-1)

        # Top 10 tokens mas probables
        top_probs, top_ids = torch.topk(probs, 10)
        print(f"\n  Top 10 tokens mas probables despues de \"{text}\":")
        print(f"  {'Rank':>4} | {'Token':>15} | {'Prob':>8} | {'Logit':>8}")
        print(f"  {'-'*4} | {'-'*15} | {'-'*8} | {'-'*8}")
        for rank, (p, tid) in enumerate(zip(top_probs, top_ids)):
            token_str = tokenizer.base_tokenizer.decode([tid.item()])
            logit_val = last_logits[tid].item()
            print(f"  {rank+1:>4} | {repr(token_str):>15} | {p.item():>7.4f} | {logit_val:>8.3f}")

        # Entropia de la distribucion
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        print(f"\n  Entropia de la distribucion: {entropy:.3f}")
        print(f"    (alta = modelo inseguro, baja = modelo confiado)")
        print(f"    Entropia maxima teorica: {torch.log(torch.tensor(float(probs.shape[0]))).item():.3f}")


# =====================================================================
# PARTE 2: ESTRATEGIAS DE SAMPLING
# =====================================================================
# El modelo produce probabilidades. Pero COMO elegimos el siguiente
# token cambia dramaticamente el resultado.
# =====================================================================

def sample_greedy(logits):
    """
    Greedy: siempre elige el token mas probable.
    Determinista. Genera texto "seguro" pero repetitivo.
    Equivalente a temperatura = 0.
    """
    return torch.argmax(logits).item()


def sample_temperature(logits, temperature):
    """
    Temperature sampling: divide logits por T antes del softmax.
    - T < 1: distribucion mas "picuda" (mas determinista)
    - T = 1: distribucion original
    - T > 1: distribucion mas "plana" (mas aleatorio)
    """
    scaled = logits / max(temperature, 1e-8)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1).item()


def sample_top_k(logits, k, temperature=1.0):
    """
    Top-k: solo considera los k tokens mas probables.
    Elimina tokens "basura" de baja probabilidad.
    GPT-2 usaba top-k=40 por defecto.
    """
    scaled = logits / max(temperature, 1e-8)
    values, indices = torch.topk(scaled, k)
    probs = F.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1).item()
    return indices[sampled_idx].item()


def sample_top_p(logits, p, temperature=1.0):
    """
    Top-p (nucleus sampling): selecciona tokens hasta que la
    probabilidad acumulada supere p.

    Adaptativo: si un token tiene 90% de probabilidad y p=0.9,
    solo se considera ESE token. Si la distribucion es plana,
    se consideran muchos tokens.

    Publicado en 2019 por Holtzman et al.
    Usado por defecto en ChatGPT/Claude.
    """
    scaled = logits / max(temperature, 1e-8)
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Encontrar el punto de corte
    cutoff_mask = cumulative_probs > p
    # Mantener al menos 1 token
    cutoff_mask[0] = False
    sorted_logits[cutoff_mask] = float('-inf')

    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1).item()
    return sorted_indices[sampled_idx].item()


def compare_sampling_strategies(model, tokenizer, prompt, n_tokens=40, device='cpu'):
    """
    Genera texto con TODAS las estrategias y compara los resultados.
    """
    model.eval()
    base_tokens = tokenizer.base_tokenizer.encode(prompt)

    strategies = {
        "Greedy (T=0)": lambda logits: sample_greedy(logits),
        "T=0.3 (conservador)": lambda logits: sample_temperature(logits, 0.3),
        "T=0.7 (balanced)": lambda logits: sample_temperature(logits, 0.7),
        "T=1.5 (creativo)": lambda logits: sample_temperature(logits, 1.5),
        "Top-k=5": lambda logits: sample_top_k(logits, 5, 0.7),
        "Top-k=40": lambda logits: sample_top_k(logits, 40, 0.7),
        "Top-p=0.5": lambda logits: sample_top_p(logits, 0.5, 0.7),
        "Top-p=0.9": lambda logits: sample_top_p(logits, 0.9, 0.7),
    }

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Generando {n_tokens} tokens con cada estrategia:")
    print()

    for name, sampler in strategies.items():
        ids = torch.tensor([base_tokens], dtype=torch.long, device=device)
        generated = []

        with torch.no_grad():
            for _ in range(n_tokens):
                if ids.shape[1] >= 128:
                    break
                logits, _ = model(ids)
                next_logits = logits[0, -1, :]
                next_id = sampler(next_logits)
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        text = tokenizer.base_tokenizer.decode(generated)
        # Limpiar para display
        text = text.replace('\n', ' ').strip()[:120]
        print(f"  {name:>22}: {text}")


# =====================================================================
# PARTE 3: TEMPERATURA VISUALIZADA
# =====================================================================

def visualize_temperature(model, tokenizer, prompt, device='cpu'):
    """
    Muestra como la temperatura cambia la distribucion de probabilidades.
    """
    model.eval()
    tokens = tokenizer.base_tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(ids)
        raw_logits = logits[0, -1, :]  # [V]

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"\n  Como la temperatura cambia la distribucion:")
    print()

    for temp in temperatures:
        scaled = raw_logits / temp
        probs = F.softmax(scaled, dim=-1)

        top_probs, top_ids = torch.topk(probs, 5)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()

        # Calcular "effective tokens" (cuantos tokens tienen prob significativa)
        effective = (probs > 0.01).sum().item()

        tokens_str = []
        for p, tid in zip(top_probs, top_ids):
            t = tokenizer.base_tokenizer.decode([tid.item()])
            tokens_str.append(f"{repr(t)}:{p.item():.3f}")

        bar_len = int(top_probs[0].item() * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)

        print(f"  T={temp:<4} | Top1: [{bar}] {top_probs[0].item():.3f} | "
              f"Entropy: {entropy:.2f} | Effective: {effective:>4} | "
              f"Top5: {', '.join(tokens_str[:3])}")

    print()
    print("  Observa:")
    print("  - T=0.1: el top-1 tiene >90% de probabilidad (casi greedy)")
    print("  - T=1.0: distribucion original del modelo")
    print("  - T=5.0: distribucion casi uniforme (puro azar)")
    print("  - Entropy sube con la temperatura (mas incertidumbre)")
    print("  - Effective tokens: cuantos tienen >1% de probabilidad")


# =====================================================================
# PARTE 4: KV-CACHE -- La optimizacion que hace posible la inferencia
# =====================================================================
# Sin KV-cache: generar N tokens requiere N forward passes,
#   y cada pass procesa TODA la secuencia (O(N^2) total).
# Con KV-cache: cada forward pass solo procesa el ULTIMO token,
#   reutilizando los K,V computados anteriormente (O(N) total).
#
# En un modelo de 70B params, esto es la diferencia entre
# 10 tokens/segundo y 100 tokens/segundo.
# =====================================================================

def generate_no_cache(model, tokenizer, prompt, n_tokens=30, device='cpu'):
    """
    Generacion SIN KV-cache: reprocesa toda la secuencia cada vez.
    Esto es lo que hicimos en Ej1/Ej2/Ej3 por simplicidad.
    """
    model.eval()
    tokens = tokenizer.base_tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_tokens):
            if ids.shape[1] >= 128:
                break
            # Procesa TODA la secuencia cada vez
            logits, _ = model(ids)
            next_logits = logits[0, -1, :]
            next_id = sample_top_k(next_logits, 20, 0.5)
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    elapsed = time.time() - t0
    text = tokenizer.base_tokenizer.decode(generated)
    tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0

    return text, elapsed, tokens_per_sec, len(generated)


def generate_with_cache(model, tokenizer, prompt, n_tokens=30, device='cpu'):
    """
    Generacion CON KV-cache: solo procesa el ultimo token.
    Los K,V de los tokens anteriores se reutilizan.

    El KV-cache ya esta implementado en nuestro modelo (SelfAttentionV2
    hereda de SelfAttention que tiene kv_cache en forward).
    """
    model.eval()
    tokens = tokenizer.base_tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = []

    t0 = time.time()
    with torch.no_grad():
        # Primer forward: procesa todo el prompt, guarda cache
        logits, kv_caches = model(ids)
        next_logits = logits[0, -1, :]
        next_id = sample_top_k(next_logits, 20, 0.5)
        generated.append(next_id)

        # Siguientes forwards: solo procesa el ultimo token
        for i in range(n_tokens - 1):
            if len(tokens) + len(generated) >= 128:
                break
            # Solo el ultimo token como input
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, kv_caches = model(next_input, kv_caches)
            next_logits = logits[0, -1, :]
            next_id = sample_top_k(next_logits, 20, 0.5)
            generated.append(next_id)

    elapsed = time.time() - t0
    text = tokenizer.base_tokenizer.decode(generated)
    tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0

    return text, elapsed, tokens_per_sec, len(generated)


def benchmark_kv_cache(model, tokenizer, device='cpu'):
    """
    Compara generacion con y sin KV-cache.
    """
    prompts = [
        "To be, or not to be",
        "The king of England",
        "In the beginning there was",
    ]

    n_tokens = 50

    print(f"\n  Benchmark: generacion de {n_tokens} tokens")
    print(f"  {'Prompt':>30} | {'Sin cache':>12} | {'Con cache':>12} | {'Speedup':>8}")
    print(f"  {'-'*30} | {'-'*12} | {'-'*12} | {'-'*8}")

    total_no_cache = 0
    total_with_cache = 0

    for prompt in prompts:
        # Sin cache
        _, t_no, tps_no, n_no = generate_no_cache(model, tokenizer, prompt, n_tokens, device)
        # Con cache
        _, t_with, tps_with, n_with = generate_with_cache(model, tokenizer, prompt, n_tokens, device)

        speedup = tps_with / tps_no if tps_no > 0 else 0
        total_no_cache += t_no
        total_with_cache += t_with

        prompt_short = prompt[:28]
        print(f"  {prompt_short:>30} | {tps_no:>8.1f} t/s | {tps_with:>8.1f} t/s | {speedup:>7.2f}x")

    overall_speedup = (total_no_cache / total_with_cache) if total_with_cache > 0 else 0
    print(f"\n  Speedup promedio: {overall_speedup:.2f}x")
    print()
    print("  Nota: en nuestro modelo de 981K params el speedup es modesto")
    print("  porque el modelo es muy pequeno. En un modelo de 7B+ params,")
    print("  el KV-cache da speedups de 5-20x porque la attention es O(n^2).")

    # Generar un ejemplo lado a lado
    prompt = "Once upon a time"
    text_no, _, _, _ = generate_no_cache(model, tokenizer, prompt, 40, device)
    text_with, _, _, _ = generate_with_cache(model, tokenizer, prompt, 40, device)
    print(f"\n  Ejemplo de generacion:")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Sin cache:  \"{text_no[:120]}\"")
    print(f"  Con cache:  \"{text_with[:120]}\"")
    print(f"  (Diferente texto porque sampling es estocastico)")


# =====================================================================
# PARTE 5: TOKEN-A-TOKEN -- ver la generacion en camara lenta
# =====================================================================

def generate_slow_motion(model, tokenizer, prompt, n_tokens=20,
                         temperature=0.7, top_k=20, device='cpu'):
    """
    Genera texto token por token mostrando las probabilidades.
    Esto es lo que pasa DENTRO de ChatGPT en cada token.
    """
    model.eval()
    tokens = tokenizer.base_tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Generando {n_tokens} tokens en camara lenta...\n")

    context = prompt
    generated_text = ""

    with torch.no_grad():
        for step in range(n_tokens):
            if ids.shape[1] >= 128:
                break

            t0 = time.time()
            logits, _ = model(ids)
            dt = (time.time() - t0) * 1000  # ms

            next_logits = logits[0, -1, :]
            raw_probs = F.softmax(next_logits, dim=-1)

            # Aplicar temperatura
            scaled = next_logits / temperature
            probs = F.softmax(scaled, dim=-1)

            # Top-k filter
            top_probs, top_ids = torch.topk(probs, min(top_k, probs.shape[0]))
            top_probs_norm = top_probs / top_probs.sum()

            # Sample
            sampled_idx = torch.multinomial(top_probs_norm, 1).item()
            next_id = top_ids[sampled_idx].item()
            chosen_prob = top_probs[sampled_idx].item()

            # Decodificar
            token_str = tokenizer.base_tokenizer.decode([next_id])
            generated_text += token_str

            # Mostrar las candidatas
            candidates = []
            for p, tid in zip(top_probs[:5], top_ids[:5]):
                t = tokenizer.base_tokenizer.decode([tid.item()])
                marker = " <--" if tid.item() == next_id else ""
                candidates.append(f"    {repr(t):>12}: {p.item():.4f}{marker}")

            entropy = -(probs * (probs + 1e-10).log()).sum().item()

            print(f"  Step {step+1:>2} | {dt:>5.1f}ms | Entropy: {entropy:.2f} | "
                  f"Elegido: {repr(token_str)} (p={chosen_prob:.4f})")
            for c in candidates:
                print(c)
            print()

            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    print(f"  Resultado final: \"{prompt}{generated_text}\"")


# =====================================================================
# PARTE 6: MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("EJERCICIO 4: Inferencia Real -- KV-Cache, Sampling, Probabilidades")
    print("=" * 70)
    print()
    print("Que hacemos en este ejercicio:")
    print("  1. Abrimos el forward pass y vemos cada capa")
    print("  2. Comparamos 8 estrategias de sampling")
    print("  3. Visualizamos como la temperatura cambia las probabilidades")
    print("  4. Implementamos KV-cache y medimos el speedup")
    print("  5. Generamos texto en 'camara lenta' (token por token)")
    print()
    print("Esto es EXACTAMENTE lo que pasa dentro de ChatGPT/Claude.")
    print("La unica diferencia es la escala (175B params vs 981K params).")
    print()

    # Cargar modelo SFT del Ej3
    ej2_dir = os.path.join(os.path.dirname(__file__), '..', 'ej2_training')
    ej3_dir = os.path.join(os.path.dirname(__file__), '..', 'ej3_mini_llm')
    tokenizer_path = os.path.join(ej2_dir, 'tokenizer_v2.json')
    model_path = os.path.join(ej3_dir, 'sft_model.pt')

    # Si no hay modelo SFT, usar el modelo base del Ej2
    if not os.path.exists(model_path):
        print("  [WARN] No se encontro sft_model.pt -- usando modelo base del Ej2")
        model_path = os.path.join(ej2_dir, 'best_model_v2.pt')

    # Config del modelo (misma que Ej2/Ej3)
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 384
    MAX_SEQ = 128
    DROPOUT = 0.0  # Sin dropout en inferencia

    chat_tokenizer = ChatTokenizer(tokenizer_path)

    model = TransformerLMv2(
        vocab_size=chat_tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        dropout=DROPOUT,
    )

    # Cargar pesos
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        state = checkpoint

    # Cargar, manejando diferencia de tamaño de vocab
    model_state = model.state_dict()
    for key in state:
        if key in model_state:
            if state[key].shape == model_state[key].shape:
                model_state[key] = state[key]
            elif 'tok_emb' in key or 'head' in key:
                # Copiar la parte que coincide (vocab expansion)
                min_size = min(state[key].shape[0], model_state[key].shape[0])
                model_state[key][:min_size] = state[key][:min_size]
    model.load_state_dict(model_state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo cargado: {n_params:,} parametros")
    print(f"  Fuente: {os.path.basename(model_path)}")
    print()

    # ------------------------------------------------------------------
    # DEMO 1: Anatomia del forward pass
    # ------------------------------------------------------------------
    print("=" * 70)
    print("DEMO 1: Anatomia de un Forward Pass")
    print("=" * 70)
    print()
    print("  Cada vez que el modelo genera UN token, pasa por estas capas.")
    print("  Mostramos los numeros reales de nuestro modelo de 981K params.")

    show_forward_pass(model, chat_tokenizer, "To be, or not to be")

    # ------------------------------------------------------------------
    # DEMO 2: Estrategias de sampling
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DEMO 2: Estrategias de Sampling")
    print("=" * 70)
    print()
    print("  Mismo prompt, 8 estrategias diferentes.")
    print("  Observa como la temperatura y el filtrado cambian el resultado.")

    compare_sampling_strategies(
        model, chat_tokenizer,
        "To be, or not to be, that is",
        n_tokens=30
    )

    compare_sampling_strategies(
        model, chat_tokenizer,
        "The king shall",
        n_tokens=30
    )

    # ------------------------------------------------------------------
    # DEMO 3: Temperatura visualizada
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DEMO 3: Efecto de la Temperatura")
    print("=" * 70)

    visualize_temperature(model, chat_tokenizer, "Romeo and Juliet")
    visualize_temperature(model, chat_tokenizer, "The soldier marched")

    # ------------------------------------------------------------------
    # DEMO 4: KV-Cache benchmark
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DEMO 4: KV-Cache -- La Optimizacion Clave")
    print("=" * 70)
    print()
    print("  Sin KV-cache: cada token nuevo requiere re-procesar")
    print("  TODA la secuencia. Con 100 tokens, el token #100")
    print("  requiere 100 forward passes de 100 tokens = 10,000 ops.")
    print()
    print("  Con KV-cache: cada token nuevo solo procesa 1 token.")
    print("  Los K,V de los tokens anteriores se guardan en memoria.")
    print("  Con 100 tokens: 100 forward passes de 1 token = 100 ops.")

    benchmark_kv_cache(model, chat_tokenizer)

    # ------------------------------------------------------------------
    # DEMO 5: Generacion en camara lenta
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DEMO 5: Generacion Token por Token (Camara Lenta)")
    print("=" * 70)
    print()
    print("  Vamos a ver EXACTAMENTE que pasa en cada paso.")
    print("  Para cada token generado, mostramos las 5 opciones")
    print("  mas probables y cual fue elegida.")

    generate_slow_motion(
        model, chat_tokenizer,
        "To be, or not to be",
        n_tokens=15,
        temperature=0.5,
        top_k=10
    )

    # ------------------------------------------------------------------
    # Explicacion final
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("LECCION DEL EJERCICIO 4")
    print("=" * 70)
    print("""
Lo que demostramos:

  1. FORWARD PASS:
     Cada token pasa por embedding -> N bloques transformer -> head.
     La salida es un vector de probabilidades sobre TODO el vocabulario.
     NO hay "pensamiento" -- solo multiplicacion de matrices.

  2. SAMPLING:
     La misma distribucion de probabilidades puede generar texto
     MUY diferente segun como elijamos:
     - Greedy: siempre el mas probable -> repetitivo
     - Temperatura baja: conservador, coherente
     - Temperatura alta: creativo, incoherente
     - Top-k: elimina tokens basura, equilibrado
     - Top-p: adaptativo, el estandar actual

  3. TEMPERATURA:
     No cambia lo que el modelo "sabe" -- solo cuanto "duda".
     T=0.1 y T=5.0 usan EXACTAMENTE los mismos pesos.
     La diferencia esta en el softmax, no en el modelo.

  4. KV-CACHE:
     Sin cache: O(n^2) operaciones para n tokens.
     Con cache: O(n) operaciones.
     En modelos grandes (7B+), esto es 5-20x mas rapido.
     Es la razon por la que los LLMs pueden responder en segundos.

  5. LA IMPLICACION PARA SEGURIDAD:
     Si entiendes que el modelo solo produce probabilidades,
     entiendes por que los "guardrails" son fragiles.
     Un jailbreak simplemente cambia las probabilidades
     para que los tokens "prohibidos" se vuelvan mas probables.
     No hay un "modulo de seguridad" separado -- todo es estadistica.

En el siguiente ejercicio construiremos TOOL CALLING desde cero:
como el modelo genera JSON para ejecutar funciones externas.
""")

    print(f"EXIT CODE: 0")


if __name__ == "__main__":
    main()
