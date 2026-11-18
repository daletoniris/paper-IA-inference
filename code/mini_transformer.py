#!/usr/bin/env python3
"""
mini_transformer.py -- Transformer decoder-only desde cero en PyTorch.
Parte 2 del paper "Hackers Artificiales o Sistemas Mal Disenados?"

Misma arquitectura que GPT. Entrenable en CPU en ~30 segundos.
Aprende a completar frases sobre agentes de IA y seguridad.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Self-attention con mascara causal (autoregresivo)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    """Attention + FFN + LayerNorm + Residual."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """Transformer decoder-only. Misma arquitectura que GPT."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new, temp=0.8, max_seq=None):
        self.eval()
        for _ in range(max_new):
            ctx = idx if max_seq is None else idx[:, -max_seq:]
            logits = self(ctx)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=-1)
        return idx


# =====================================================================
# DATASET Y ENTRENAMIENTO
# =====================================================================

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
    "el modelo genera texto y el runtime decide que ejecutar",
    "la seguridad no esta en el prompt esta en el codigo",
    "un perceptron es una neurona artificial simple",
    "self attention permite ver toda la secuencia al mismo tiempo",
    "el entrenamiento ajusta millones de parametros",
]

# Vocabulario
words = set()
for t in textos:
    words.update(t.split())

w2i = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
for i, w in enumerate(sorted(words), 3):
    w2i[w] = i
i2w = {v: k for k, v in w2i.items()}
V = len(w2i)

# Tokenizar
def tok(text):
    return [w2i["<bos>"]] + [w2i[w] for w in text.split()] + [w2i["<eos>"]]

seqs = [tok(t) for t in textos]
ML = max(len(s) for s in seqs)
for s in seqs:
    s.extend([0] * (ML - len(s)))

data = torch.tensor(seqs)

# Hiperparametros
D = 64
H = 4
L = 3
FF = 128

model = MiniGPT(V, D, H, L, FF, ML)
n_params = sum(p.numel() for p in model.parameters())

print("=" * 60)
print("MINI-TRANSFORMER (arquitectura GPT)")
print(f"Vocabulario:  {V} tokens")
print(f"Dimensiones:  d_model={D}, heads={H}, layers={L}, ff={FF}")
print(f"Parametros:   {n_params:,}")
print(f"(GPT-4 tiene ~1,800,000,000,000)")
print("=" * 60)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
EPOCHS = 500

for epoch in range(EPOCHS):
    model.train()
    inp = data[:, :-1]
    tgt = data[:, 1:]
    logits = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoca {epoch+1:>4}: loss = {loss.item():.4f}")

# Generacion
print()
print("=" * 60)
print("GENERACION DE TEXTO (token por token, como GPT)")
print("=" * 60)

for start in ["el", "la", "un", "los"]:
    if start in w2i:
        seed = torch.tensor([[w2i["<bos>"], w2i[start]]])
        out = model.generate(seed, max_new=8, temp=0.5, max_seq=ML-1)
        words_out = [i2w.get(t, "?") for t in out[0].tolist()]
        # Stop at <eos> if present
        if "<eos>" in words_out:
            words_out = words_out[:words_out.index("<eos>") + 1]
        print(f"  '{start}' -> {' '.join(words_out)}")

print()
print("Cada token fue generado UNO A LA VEZ, de izquierda a derecha.")
print("No hay 'pensamiento'. Es prediccion estadistica del siguiente token.")
print(f"Este modelo tiene {n_params:,} parametros. GPT-4 tiene 1.8 trillones.")
print("La arquitectura es la misma. La diferencia es escala y datos.")
