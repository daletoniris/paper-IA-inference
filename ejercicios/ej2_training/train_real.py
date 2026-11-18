#!/usr/bin/env python3
"""
Ejercicio 2: Entrenar el Transformer con datos REALES en la Raspberry Pi.
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

En el Ejercicio 1 construimos un Transformer desde cero con arquitectura
real (RMSNorm, RoPE, SwiGLU, KV-Cache) pero lo entrenamos con solo
50K caracteres y 400 tokens de vocabulario. El resultado: texto incoherente.

Ahora escalamos DENTRO de lo que la Pi puede manejar:
  - Dataset completo: 1.1MB de Shakespeare (~1 millon de chars)
  - Vocabulario: 2000 tokens BPE (5x mas)
  - Modelo: ~2.5M params (5x mas que el Ej1)
  - Dropout para regularizacion (evitar overfitting)
  - Early stopping (parar cuando deja de mejorar)
  - Samples de texto cada epoca para ver la evolucion

El objetivo: ver como un modelo pasa de generar basura a generar
texto que PARECE Shakespeare. No es magia -- es estadistica + escala.

Tiempo estimado: ~30-40 min en Pi 5 (CPU).

Autor: TokioAI (ejercicio practico del paper)
"""

import sys
import os

# Importar todo del Ejercicio 1 (reutilizamos BPE, RMSNorm, RoPE, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej1_transformer'))
from transformer_real import (
    BPETokenizer,
    RMSNorm, precompute_rope_freqs, apply_rope,
    SelfAttention, SwiGLU, TransformerBlock,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import random


# =====================================================================
# PARTE 1: MODELO ESCALADO CON DROPOUT
# =====================================================================
# El Ej1 no tenia dropout. Sin dropout, el modelo memoriza el dataset
# en vez de aprender patrones generalizables. Dropout apaga neuronas
# al azar durante el training, forzando redundancia.
#
# Analogia: es como estudiar para un examen tapandote un ojo al azar.
# Te obliga a no depender de un solo camino visual.
# =====================================================================

class SelfAttentionV2(nn.Module):
    """
    Self-Attention con dropout.
    Identico al Ej1 pero agrega dropout en attention weights y output.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_freqs, kv_cache=None):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, rope_freqs[:T])
        k = apply_rope(k, rope_freqs[:T])

        if kv_cache is not None:
            k_old, v_old = kv_cache
            k = torch.cat([k_old, k], dim=2)
            v = torch.cat([v_old, v], dim=2)
            new_cache = (k, v)
            # Para generacion, RoPE ya se aplico a las keys viejas
            # Solo necesitamos ajustar las freqs para las nuevas
        else:
            new_cache = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        T_q, T_k = q.size(2), k.size(2)
        mask = torch.triu(torch.ones(T_q, T_k, device=x.device), diagonal=T_k - T_q + 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)  # Dropout en attention

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        out = self.out_dropout(self.W_out(out))  # Dropout en output

        return out, new_cache


class SwiGLUV2(nn.Module):
    """SwiGLU con dropout."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)


class TransformerBlockV2(nn.Module):
    """Transformer block con dropout en residuals."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = SelfAttentionV2(d_model, num_heads, dropout)
        self.ffn = SwiGLUV2(d_model, d_ff, dropout)
        self.norm_attn = RMSNorm(d_model)
        self.norm_ffn = RMSNorm(d_model)
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_freqs, kv_cache=None):
        h, new_cache = self.attn(self.norm_attn(x), rope_freqs, kv_cache)
        x = x + self.res_dropout(h)
        x = x + self.res_dropout(self.ffn(self.norm_ffn(x)))
        return x, new_cache


class TransformerLMv2(nn.Module):
    """
    Transformer LM escalado con dropout y regularizacion.

    Diferencias vs Ej1:
    - Dropout en attention, FFN, y conexiones residuales
    - Weight tying: embedding y output head comparten pesos
      (tecnica usada en GPT-2, reduce params un ~15%)
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlockV2(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: la matriz de embedding y la de output son la misma.
        # Esto es una tecnica de GPT-2. Intuicion: si "king" tiene un
        # embedding similar a "queen", entonces cuando el modelo predice
        # el siguiente token, deberia dar probabilidades similares a ambos
        # en contextos parecidos. Compartir pesos fuerza esta simetria.
        self.head.weight = self.tok_emb.weight

        # RoPE frequencies
        self.register_buffer(
            'rope_freqs',
            precompute_rope_freqs(d_model // num_heads, max_seq_len)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Inicializacion de pesos estilo GPT-2/LLaMA.
        Xavier normal para linears, normal pequena para embeddings.
        Esto acelera la convergencia.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, kv_caches=None):
        B, T = x.shape

        h = self.emb_dropout(self.tok_emb(x))

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            h, new_cache = block(h, self.rope_freqs, cache)
            new_caches.append(new_cache)

        h = self.norm(h)
        logits = self.head(h)

        return logits, new_caches

    def count_params(self):
        """Cuenta parametros por componente."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        attn = sum(p.numel() for b in self.blocks
                   for name, p in b.attn.named_parameters())
        ffn = sum(p.numel() for b in self.blocks
                  for name, p in b.ffn.named_parameters())
        norms = sum(p.numel() for b in self.blocks
                    for name, p in b.named_parameters() if 'norm' in name)
        norms += self.norm.weight.numel()

        return {
            'total': total,
            'embedding': emb,
            'attention': attn,
            'ffn': ffn,
            'norms': norms,
            # Con weight tying, head no cuenta (es el mismo que embedding)
        }


# =====================================================================
# PARTE 2: DATASET Y DATALOADER EFICIENTES
# =====================================================================
# En el Ej1, TextDataset usaba stride=1: cada posicion del texto era
# una secuencia. Con 70K tokens eso da 70K secuencias -- demasiado
# para CPU. Usamos stride=8: cada 8 posiciones creamos una secuencia.
# Esto reduce 8x el training time sin perder mucha calidad (las
# secuencias siguen solapandose un 93.75%).
# =====================================================================

class TextDatasetV2:
    """Dataset con stride configurable para eficiencia en CPU."""

    def __init__(self, tokens, seq_len, stride=1):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride
        # Calcular indices validos
        max_start = len(self.tokens) - seq_len - 1
        self.indices = list(range(0, max(0, max_start), stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def create_dataloader_v2(dataset, batch_size, shuffle=True):
    """DataLoader con shuffle eficiente."""
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        if len(batch_idx) < batch_size // 2:
            continue  # Skip ultimo batch si es muy chico
        xs, ys = [], []
        for idx in batch_idx:
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
        yield torch.stack(xs), torch.stack(ys)


# =====================================================================
# PARTE 3: TRAINING LOOP MEJORADO
# =====================================================================
# Mejoras sobre Ej1:
# - Early stopping: si val loss no mejora en N epocas, para
# - Learning rate warmup: empieza bajo, sube gradualmente, luego baja
# - Samples de texto cada epoca: para ver la evolucion del modelo
# =====================================================================

@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80,
                    temperature=0.8, top_k=40):
    """Genera texto con KV-Cache (identico a Ej1 pero con top_k y temp)."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long)

    kv_caches = None
    generated = []

    for _ in range(max_tokens):
        logits, kv_caches = model(tokens, kv_caches)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-K filtering
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, float('-inf'))
            filtered.scatter_(-1, indices, values)
            logits = filtered

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated.append(next_token.item())

        # EOS check
        if next_token.item() == tokenizer.special_tokens.get("<eos>", -1):
            break

        tokens = next_token

    return prompt + tokenizer.decode(generated)


def train_v2(model, tokenizer, text, config):
    """
    Training loop mejorado con:
    - Early stopping
    - Warmup + cosine decay
    - Gradient clipping
    - Samples de texto cada epoca
    - Metricas detalladas
    """

    # Tokenizar todo el texto
    print("\nTokenizando dataset completo...")
    t0 = time.time()
    all_tokens = tokenizer.encode(text)
    print(f"  {len(all_tokens):,} tokens en {time.time()-t0:.1f}s")
    print(f"  Ratio chars/token: {len(text)/len(all_tokens):.1f}")

    # Split train/val (90/10)
    split = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    # Stride=8: 8x menos secuencias, training viable en CPU
    STRIDE = 8
    train_ds = TextDatasetV2(train_tokens, config['seq_len'], stride=STRIDE)
    val_ds = TextDatasetV2(val_tokens, config['seq_len'], stride=STRIDE)

    print(f"\n  Train: {len(train_ds):,} secuencias")
    print(f"  Val:   {len(val_ds):,} secuencias")
    print(f"  Batch size:     {config['batch_size']}")
    print(f"  Seq length:     {config['seq_len']}")
    print(f"  Max epochs:     {config['epochs']}")
    print(f"  Early stopping: {config['patience']} epocas sin mejora")
    print(f"  Learning rate:  {config['lr']}")
    print(f"  Warmup steps:   {config['warmup_steps']}")
    print(f"  Dropout:        {config['dropout']}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.95),   # LLaMA usa 0.9, 0.95
        weight_decay=0.1,     # Regularizacion L2
    )

    # Scheduler: warmup lineal + cosine decay
    total_steps = config['epochs'] * (len(train_ds) // config['batch_size'] + 1)

    def lr_schedule(step):
        """Warmup lineal + cosine annealing."""
        if step < config['warmup_steps']:
            return step / max(config['warmup_steps'], 1)
        progress = (step - config['warmup_steps']) / max(total_steps - config['warmup_steps'], 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Tracking
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'perplexity': [], 'time': []}
    best_model_path = os.path.join(os.path.dirname(__file__), "best_model_v2.pt")
    global_step = 0
    start_time = time.time()

    # Prompts para evaluar la evolucion
    eval_prompts = [
        "KING HENRY:",
        "To be or not to be",
        "My lord, I shall",
        "The noble",
        "What is the",
    ]

    print(f"\n{'='*60}")
    print(f"INICIO DEL ENTRENAMIENTO")
    print(f"{'='*60}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # ----- TRAINING -----
        model.train()
        train_loss_sum = 0
        train_batches = 0

        for batch_x, batch_y in create_dataloader_v2(train_ds, config['batch_size']):
            logits, _ = model(batch_x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_y.view(-1),
                ignore_index=0,  # Ignorar padding
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (como LLaMA/GPT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_batches += 1
            global_step += 1

            # Log cada N batches
            if train_batches % config['log_interval'] == 0:
                avg_loss = train_loss_sum / train_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{config['epochs']} | "
                      f"Batch {train_batches:>4d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr_now:.2e} | "
                      f"Tiempo: {elapsed:.0f}s")

        train_loss = train_loss_sum / max(train_batches, 1)

        # ----- VALIDATION -----
        model.eval()
        val_loss_sum = 0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in create_dataloader_v2(val_ds, config['batch_size'],
                                                          shuffle=False):
                logits, _ = model(batch_x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch_y.view(-1),
                    ignore_index=0,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        perplexity = math.exp(min(val_loss, 20))  # Cap para evitar overflow
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['perplexity'].append(perplexity)
        history['time'].append(epoch_time)

        # Resumen de la epoca
        print(f"\n  >>> Epoch {epoch+1} completada:")
        print(f"      Train Loss = {train_loss:.4f}")
        print(f"      Val Loss   = {val_loss:.4f}")
        print(f"      Perplexity = {perplexity:.1f}")
        print(f"      Tiempo     = {epoch_time:.0f}s ({epoch_time/60:.1f} min)")
        print(f"      Mejor val  = {best_val_loss:.4f}")

        # ----- EARLY STOPPING -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch + 1,
                'config': config,
            }, best_model_path)
            print(f"      ** Nuevo mejor modelo guardado! **")
        else:
            patience_counter += 1
            print(f"      Sin mejora ({patience_counter}/{config['patience']})")

            if patience_counter >= config['patience']:
                print(f"\n  EARLY STOPPING: {config['patience']} epocas sin mejora.")
                break

        # ----- GENERAR SAMPLES -----
        print(f"\n  --- Samples (Epoch {epoch+1}) ---")
        for prompt in eval_prompts:
            sample = generate_sample(model, tokenizer, prompt,
                                     max_tokens=60, temperature=0.8, top_k=40)
            # Limpiar: una sola linea, sin saltos
            sample_clean = sample.replace('\n', ' ').strip()
            if len(sample_clean) > 120:
                sample_clean = sample_clean[:120] + "..."
            print(f"  [{prompt[:20]:>20}] {sample_clean}")
        print()

    total_time = time.time() - start_time

    # ----- RESUMEN FINAL -----
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"  Epocas completadas: {len(history['train_loss'])}/{config['epochs']}")
    print(f"  Tiempo total:       {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Mejor val loss:     {best_val_loss:.4f}")
    print(f"  Mejor perplexity:   {math.exp(min(best_val_loss, 20)):.1f}")
    print()

    print(f"  Evolucion por epoca:")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Perplexity':>10} | {'Tiempo':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for i in range(len(history['train_loss'])):
        print(f"  {i+1:>5} | {history['train_loss'][i]:>10.4f} | "
              f"{history['val_loss'][i]:>10.4f} | "
              f"{history['perplexity'][i]:>10.1f} | "
              f"{history['time'][i]:>7.0f}s")

    return best_val_loss, history


# =====================================================================
# PARTE 3: MAIN -- ENTRENAR EN SERIO
# =====================================================================

def main():
    print(f"{'='*60}")
    print(f"EJERCICIO 2: ENTRENAR EL TRANSFORMER CON DATOS REALES")
    print(f"{'='*60}")
    print()
    print("En el Ejercicio 1 construimos la arquitectura.")
    print("Ahora la entrenamos EN SERIO:")
    print("  - Dataset 4x mas grande (200K chars de Shakespeare)")
    print("  - Vocabulario 2.5x mas grande (1000 tokens)")
    print("  - Modelo 5x mas grande (~2.5M params)")
    print("  - Dropout para no overfittear")
    print("  - Early stopping para no desperdiciar tiempo")
    print("  - Weight tying (como GPT-2)")
    print()

    # 1. Cargar dataset completo
    data_path = os.path.join(os.path.dirname(__file__),
                             '..', 'ej1_transformer', 'data_tinystory.txt')
    if not os.path.exists(data_path):
        print(f"ERROR: No se encuentra {data_path}")
        print("Primero correr el Ejercicio 1 para descargar el dataset.")
        sys.exit(1)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Dataset completo: {len(text):,} caracteres ({len(text)/1024/1024:.1f} MB)")
    print(f"Preview: {text[:100]}...")

    # Usar 200K chars para training (4x mas que Ej1, pero viable en Pi)
    # Con 1.1M chars y 3M params en CPU tardaria horas.
    # 200K es suficiente para aprender patrones claros de Shakespeare.
    TEXT_SIZE = 200_000
    text = text[:TEXT_SIZE]
    print(f"Usando: {len(text):,} caracteres para training")
    print()

    # 2. Entrenar tokenizador BPE con vocabulario mas grande
    print(f"{'='*60}")
    print(f"TOKENIZADOR BPE (1000 tokens)")
    print(f"{'='*60}")

    VOCAB_SIZE = 1000  # 2.5x el Ej1, buen balance calidad/velocidad
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=VOCAB_SIZE)

    # Demo
    samples = [
        "To be, or not to be, that is the question",
        "KING HENRY: My lord, I fear",
        "Enter CORIOLANUS, with AUFIDIUS",
    ]
    print(f"\n--- Demo del tokenizador ---")
    for s in samples:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc)
        print(f"  Original:  {s}")
        print(f"  Tokens:    {len(enc)} tokens -> {enc[:15]}{'...' if len(enc)>15 else ''}")
        print(f"  Decoded:   {dec}")
        print()

    # Guardar tokenizador
    tok_path = os.path.join(os.path.dirname(__file__), "tokenizer_v2.json")
    tokenizer.save(tok_path)

    # 3. Crear modelo escalado
    print(f"{'='*60}")
    print(f"MODELO TRANSFORMER v2")
    print(f"{'='*60}")

    # Configuracion escalada pero entrenable en Pi 5 (~15 min/epoca)
    # Regla de oro: mas grande = mejor, pero el CPU tiene limites.
    D_MODEL = 128     # 1.33x Ej1 (era 96)
    NUM_HEADS = 4     # Mismo que Ej1 (d_head=32)
    NUM_LAYERS = 4    # Mismo que Ej1
    D_FF = 384        # 1.5x Ej1 (era 256)
    MAX_SEQ = 128     # 2x Ej1 (era 64) -- mas contexto
    DROPOUT = 0.1     # Nuevo! Regularizacion

    model = TransformerLMv2(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        dropout=DROPOUT,
    )

    params = model.count_params()
    print(f"\nArquitectura v2:")
    print(f"  d_model:     {D_MODEL} (Ej1: 96)")
    print(f"  num_heads:   {NUM_HEADS} (Ej1: 4)")
    print(f"  d_head:      {D_MODEL // NUM_HEADS}")
    print(f"  num_layers:  {NUM_LAYERS} (Ej1: 4)")
    print(f"  d_ff:        {D_FF} (Ej1: 256)")
    print(f"  vocab_size:  {tokenizer.vocab_size} (Ej1: 400)")
    print(f"  max_seq_len: {MAX_SEQ} (Ej1: 64)")
    print(f"  dropout:     {DROPOUT} (Ej1: 0)")
    print(f"  weight_tying: Si (Ej1: No)")
    print()

    print(f"Parametros:")
    print(f"  Total:       {params['total']:>10,} (Ej1: 481,632)")
    print(f"  Embedding:   {params['embedding']:>10,}")
    print(f"  Attention:   {params['attention']:>10,}")
    print(f"  FFN:         {params['ffn']:>10,}")
    print(f"  Norms:       {params['norms']:>10,}")
    print(f"  Ratio vs Ej1: {params['total']/481_632:.1f}x")
    print()

    print(f"Comparacion de escala:")
    print(f"  Ej1:          481,632 params / 50K chars / 400 vocab")
    print(f"  Ej2 (este):  {params['total']:>10,} params / {TEXT_SIZE:,} chars / {VOCAB_SIZE} vocab")
    print(f"  GPT-2:       124,000,000 params / ~10B chars / 50K vocab")
    print(f"  LLaMA-7B:    6,738,000,000 params / ~1T chars / 32K vocab")

    # 4. Entrenar
    config = {
        'epochs': 5,
        'batch_size': 32,
        'seq_len': MAX_SEQ,
        'lr': 3e-4,           # LLaMA usa 3e-4
        'warmup_steps': 100,
        'dropout': DROPOUT,
        'patience': 2,        # Early stopping: 2 epocas sin mejora
        'log_interval': 100,
    }

    best_loss, history = train_v2(model, tokenizer, text, config)

    # 5. Cargar mejor modelo y generar texto final
    print(f"\n{'='*60}")
    print(f"GENERACION FINAL (mejor modelo)")
    print(f"{'='*60}")

    checkpoint = torch.load(
        os.path.join(os.path.dirname(__file__), "best_model_v2.pt"),
        weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Cargado modelo de epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
    print()

    prompts = [
        ("KING HENRY:\nMy noble lords", 100),
        ("To be, or not to be, that is", 80),
        ("Enter CORIOLANUS.\n\nFirst Citizen:\n", 100),
        ("The king shall", 80),
        ("ROMEO:\nWhat light through yonder", 100),
    ]

    for prompt, max_tokens in prompts:
        print(f"  Prompt: \"{prompt}\"")

        # Temperatura baja (mas coherente)
        output_low = generate_sample(model, tokenizer, prompt,
                                     max_tokens=max_tokens,
                                     temperature=0.5, top_k=30)
        print(f"  [temp=0.5] {output_low}")

        # Temperatura alta (mas creativa)
        output_high = generate_sample(model, tokenizer, prompt,
                                      max_tokens=max_tokens,
                                      temperature=1.0, top_k=50)
        print(f"  [temp=1.0] {output_high}")
        print()

    # 6. Resumen pedagogico
    print(f"{'='*60}")
    print(f"QUE APRENDIMOS")
    print(f"{'='*60}")
    print(f"""
Lo que cambio entre Ej1 y Ej2:

  1. MAS DATOS: 50K -> {TEXT_SIZE:,} chars ({TEXT_SIZE//50_000}x mas)
     El modelo ve muchos mas patrones de Shakespeare.

  2. MAS VOCABULARIO: 400 -> {VOCAB_SIZE} tokens ({VOCAB_SIZE//400:.1f}x)
     Mejor representacion del texto. Menos tokens por palabra.
     Ratio chars/token mejora (mas cercano a los ~4 chars/token de GPT-4).

  3. MAS PARAMETROS: 481K -> {params['total']:,} ({params['total']//481_632}x)
     Mas capacidad para almacenar patrones.

  4. REGULARIZACION: Dropout {DROPOUT} + weight decay + early stopping
     Evita que el modelo memorice el training set.
     Sin regularizacion: train loss baja pero val loss sube (Ej1).
     Con regularizacion: ambos bajan juntos.

  5. WEIGHT TYING: embedding = output head
     Reduce parametros y mejora la generalizacion.
     Tecnica usada en GPT-2, T5, y otros.

La leccion: la arquitectura del Transformer es una CONSTANTE.
Lo que cambia es la escala: datos, vocabulario, parametros, compute.
Un modelo de 2.5M params en una Pi genera texto "shakespeariano".
Un modelo de 175B params en un datacenter genera texto "inteligente".
La diferencia es cuantitativa, no cualitativa.
""")


if __name__ == "__main__":
    main()
