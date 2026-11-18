#!/usr/bin/env python3
"""
Ejercicio 3: De Transformer a Mini-LLM -- Supervised Fine-Tuning (SFT)
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

En el Ej1 construimos un Transformer desde cero (RMSNorm, RoPE, SwiGLU).
En el Ej2 lo entrenamos con Shakespeare -- aprendio a COMPLETAR texto.

Pero completar texto != seguir instrucciones.
Un modelo que completa "What is TCP?" podria continuar con "What is UDP?"
en vez de RESPONDER la pregunta.

Este ejercicio demuestra EXACTAMENTE como se transforma un "completador
de texto" en un "asistente" via SFT (Supervised Fine-Tuning):

  1. Tomamos el modelo pre-entrenado del Ej2 (981K params, Shakespeare)
  2. Construimos un dataset de instruccion-respuesta en formato CHAT
  3. Usamos un chat template (como los LLMs reales: <|user|>, <|assistant|>)
  4. Fine-tuneamos SOLO las ultimas capas (para no destruir lo aprendido)
  5. Comparamos: modelo base (completa texto) vs modelo SFT (sigue instrucciones)

La leccion clave: el SFT NO le "ensena conocimiento" al modelo.
Le ensena un FORMATO de respuesta. El conocimiento viene del pre-training.
Por eso un modelo SFT con 981K params no va a "saber" cosas --
pero SI va a intentar responder en formato pregunta-respuesta.

Tiempo estimado: ~15-20 min en Pi 5 (CPU).

Autor: TokioAI (ejercicio practico del paper)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import random
import copy

# Reutilizamos TODA la arquitectura del Ej1 + Ej2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej1_transformer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ej2_training'))
from transformer_real import (
    BPETokenizer,
    RMSNorm, precompute_rope_freqs, apply_rope,
)
from train_real import (
    SelfAttentionV2, SwiGLUV2, TransformerBlockV2, TransformerLMv2,
)


# =====================================================================
# PARTE 1: CHAT TEMPLATE -- el "formato" que convierte texto en dialogo
# =====================================================================
# Los LLMs modernos usan tokens especiales para separar roles:
#   - Qwen:    <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
#   - LLaMA:   <|begin_of_text|><|start_header_id|>user<|end_header_id|>...
#   - Mistral: [INST] ... [/INST]
#   - GPT:     (via API, no tokens visibles, pero internamente similares)
#
# Nosotros usamos un formato simple pero real:
#   <|system|> {system_prompt} <|end|>
#   <|user|> {user_message} <|end|>
#   <|assistant|> {assistant_response} <|end|>
#
# El modelo aprende que despues de <|user|>...<|end|><|assistant|>
# debe GENERAR una respuesta, no continuar el texto del usuario.
# =====================================================================

# Tokens especiales de nuestro chat template
SPECIAL_TOKENS = {
    "<|system|>": None,      # IDs se asignan al construir tokenizer
    "<|user|>": None,
    "<|assistant|>": None,
    "<|end|>": None,
    "<|pad|>": None,
}


class ChatTokenizer:
    """
    Extiende el BPE tokenizer del Ej1 con tokens especiales para chat.

    Los LLMs reales hacen exactamente esto:
    - Entrenan BPE en texto plano (pre-training)
    - Agregan tokens especiales al vocabulario (SFT)
    - El embedding se expande para incluir los nuevos tokens

    Nosotros hacemos lo mismo: cargamos el tokenizer del Ej2
    y le agregamos <|system|>, <|user|>, <|assistant|>, <|end|>.
    """

    def __init__(self, base_tokenizer_path):
        # Cargar tokenizer base del Ej2
        self.base_tokenizer = BPETokenizer()  # will be overwritten by load()
        self.base_tokenizer.load(base_tokenizer_path)

        # Vocabulario base
        self.base_vocab_size = self.base_tokenizer.vocab_size

        # Agregar tokens especiales
        self.special_tokens = {}
        for i, token in enumerate(SPECIAL_TOKENS.keys()):
            token_id = self.base_vocab_size + i
            self.special_tokens[token] = token_id

        # Vocabulario total
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)

        # IDs rapidos
        self.system_id = self.special_tokens["<|system|>"]
        self.user_id = self.special_tokens["<|user|>"]
        self.assistant_id = self.special_tokens["<|assistant|>"]
        self.end_id = self.special_tokens["<|end|>"]
        self.pad_id = self.special_tokens["<|pad|>"]

    def encode_chat(self, messages):
        """
        Convierte una lista de mensajes en tokens.

        Input:
          [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is TCP?"},
            {"role": "assistant", "content": "TCP is a protocol..."},
          ]

        Output:
          [<|system|>] + encode("You are helpful.") + [<|end|>] +
          [<|user|>]   + encode("What is TCP?")     + [<|end|>] +
          [<|assistant|>] + encode("TCP is a protocol...") + [<|end|>]

        Esto es EXACTAMENTE lo que hacen los LLMs reales.
        La unica diferencia es la cantidad de tokens especiales.
        """
        token_ids = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Token de inicio de rol
            if role == "system":
                token_ids.append(self.system_id)
            elif role == "user":
                token_ids.append(self.user_id)
            elif role == "assistant":
                token_ids.append(self.assistant_id)

            # Contenido tokenizado con BPE
            content_ids = self.base_tokenizer.encode(content)
            token_ids.extend(content_ids)

            # Token de fin
            token_ids.append(self.end_id)

        return token_ids

    def encode_chat_for_generation(self, messages):
        """
        Como encode_chat, pero SIN el <|end|> del ultimo mensaje assistant.
        Se usa durante la generacion: el modelo debe generar la respuesta.

        Termina en <|assistant|> para que el modelo complete la respuesta.
        """
        token_ids = []

        for msg in messages[:-1]:  # Todos menos el ultimo
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                token_ids.append(self.system_id)
            elif role == "user":
                token_ids.append(self.user_id)
            elif role == "assistant":
                token_ids.append(self.assistant_id)

            content_ids = self.base_tokenizer.encode(content)
            token_ids.extend(content_ids)
            token_ids.append(self.end_id)

        # Ultimo mensaje: solo el token de rol (el modelo genera el resto)
        last = messages[-1]
        if last["role"] == "assistant":
            token_ids.append(self.assistant_id)
        elif last["role"] == "user":
            token_ids.append(self.user_id)

        # Si el ultimo tiene contenido parcial, agregarlo
        if last.get("content"):
            content_ids = self.base_tokenizer.encode(last["content"])
            token_ids.extend(content_ids)

        return token_ids

    def decode(self, token_ids):
        """Decodifica tokens incluyendo especiales."""
        result = []
        base_ids = []

        for tid in token_ids:
            if tid >= self.base_vocab_size:
                # Primero decodificar tokens base acumulados
                if base_ids:
                    result.append(self.base_tokenizer.decode(base_ids))
                    base_ids = []
                # Token especial
                for name, sid in self.special_tokens.items():
                    if sid == tid:
                        result.append(name)
                        break
            else:
                base_ids.append(tid)

        if base_ids:
            result.append(self.base_tokenizer.decode(base_ids))

        return "".join(result)


# =====================================================================
# PARTE 2: DATASET SFT -- Instrucciones en "Shakespearean"
# =====================================================================
# Problema: nuestro modelo solo conoce Shakespeare.
# No podemos pedirle que explique TCP -- nunca vio esa palabra.
#
# Solucion: creamos instrucciones que el modelo PUEDE responder
# usando el conocimiento que ya tiene (vocabulario Shakespeare).
#
# Esto es analogo a lo que pasa en los LLMs reales:
# - Pre-training: el modelo aprende TODO el conocimiento de internet
# - SFT: le ensenan a RESPONDER en formato pregunta-respuesta
# - El SFT no agrega conocimiento nuevo, solo formato
#
# En nuestro caso el "conocimiento" es limitado (solo Shakespeare),
# pero el principio es el mismo.
# =====================================================================

SFT_DATASET = [
    # --- Formato basico: pregunta -> respuesta ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Who is Romeo?"},
            {"role": "assistant", "content": "Romeo is a young man of the house of Montague, who falls in love with Juliet of the house of Capulet. Their love is forbidden."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Who is Hamlet?"},
            {"role": "assistant", "content": "Hamlet is the Prince of Denmark, son of the murdered King. He seeks revenge against his uncle Claudius who killed his father and married his mother."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is the play about?"},
            {"role": "assistant", "content": "The play is a tale of love, death, honour, and the fate of kings and lords. It speaks of the human heart and its many sorrows."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about Macbeth."},
            {"role": "assistant", "content": "Macbeth is a brave lord of Scotland who is told by three witches that he shall be king. Driven by ambition and his wife, he murders King Duncan and takes the throne."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What happens to Juliet?"},
            {"role": "assistant", "content": "Juliet takes a potion that makes her seem dead. Romeo, believing her dead, takes poison. When Juliet wakes and finds Romeo dead, she takes his dagger and dies."},
        ]
    },
    # --- Preguntas mas cortas / directas ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Name a villain."},
            {"role": "assistant", "content": "Iago, from Othello. He is a most cunning and false villain who destroys the noble Moor through lies and treachery."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is love?"},
            {"role": "assistant", "content": "Love is not love which alters when it alteration finds. It is an ever-fixed mark that looks on tempests and is never shaken."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Quote something famous."},
            {"role": "assistant", "content": "To be, or not to be, that is the question. Whether it is nobler in the mind to suffer the slings and arrows of outrageous fortune."},
        ]
    },
    # --- Multi-turn (el modelo ve contexto previo) ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about a king."},
            {"role": "assistant", "content": "King Lear is an old king who divides his kingdom among his three daughters. The two elder daughters betray him, and only the youngest, Cordelia, truly loves him."},
            {"role": "user", "content": "What happens to him?"},
            {"role": "assistant", "content": "Lear goes mad upon the heath, cast out by his own daughters. In the end, Cordelia dies in his arms, and Lear dies of grief. It is the most sorrowful of all the plays."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Who is Prospero?"},
            {"role": "assistant", "content": "Prospero is the rightful Duke of Milan, cast away to an island with his daughter Miranda. He commands spirits and magic, chief among them the spirit Ariel."},
            {"role": "user", "content": "Does he get revenge?"},
            {"role": "assistant", "content": "He brings his enemies to the island by tempest, but in the end he chooses mercy over revenge. He breaks his staff and drowns his book of magic."},
        ]
    },
    # --- Instrucciones creativas ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Write a short verse."},
            {"role": "assistant", "content": "O gentle night, that wraps the world in sleep,\nWhere lovers dream and silent waters weep,\nThe moon doth cast her silver on the ground,\nAnd not a soul doth make a sound."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Describe the night."},
            {"role": "assistant", "content": "The night is dark and full of whispers. The stars are like a thousand eyes that watch the deeds of men. In the silence, the world holds its breath."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What makes a good king?"},
            {"role": "assistant", "content": "A good king is just and merciful. He listens to his people and does not let pride govern his heart. He knows that the crown is a burden, not a prize."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about death."},
            {"role": "assistant", "content": "Death is the undiscovered country from whose bourn no traveller returns. It is the end that all men fear, yet none can escape. In Shakespeare, death comes for kings and fools alike."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Speak of honour."},
            {"role": "assistant", "content": "Honour is the soldier's bread. Without it, a man is nothing but a shadow. Yet honour can be a trap, for many have died in its pursuit while the dishonourable prosper."},
        ]
    },
    # --- Preguntas que requieren "razonamiento" (patron aprendido) ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Why do the plays still matter?"},
            {"role": "assistant", "content": "Because they speak of things that never change: love, jealousy, ambition, fear, and the longing for meaning. The words are old, but the hearts they describe are the same as ours."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Compare Romeo and Hamlet."},
            {"role": "assistant", "content": "Romeo acts too quickly, ruled by passion. Hamlet acts too slowly, paralyzed by thought. Both are destroyed. The lesson is that neither extreme serves a man well."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is a tragedy?"},
            {"role": "assistant", "content": "A tragedy is a tale where a great person falls from high to low through some flaw in their nature. Pride, jealousy, indecision -- these are the seeds of ruin."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is a comedy?"},
            {"role": "assistant", "content": "A comedy is a play that ends in marriage and joy. There are disguises, mistaken identities, and lovers who quarrel before they unite. The world is set right in the end."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Who is the fool?"},
            {"role": "assistant", "content": "The fool is the wisest man in the court. He speaks truth to power through jest and riddle. In King Lear, the Fool sees what the king cannot: that his daughters are false."},
        ]
    },
    # --- Mas datos para el modelo ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about Othello."},
            {"role": "assistant", "content": "Othello is a noble Moor and general who is deceived by Iago into believing his wife Desdemona is unfaithful. Consumed by jealousy, he murders her and then takes his own life upon learning the truth."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is the meaning of the tempest?"},
            {"role": "assistant", "content": "The Tempest is about power and forgiveness. Prospero could destroy his enemies but chooses to forgive them. It is Shakespeare's farewell to the stage and to the magic of art."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Describe a battle."},
            {"role": "assistant", "content": "Once more unto the breach, dear friends! The field runs red with blood, swords clash upon shields, and men cry out for glory or for mercy. War is the sport of kings and the grave of common men."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What does Portia say about mercy?"},
            {"role": "assistant", "content": "The quality of mercy is not strained. It droppeth as the gentle rain from heaven upon the place beneath. It is twice blessed: it blesses him that gives and him that takes."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about the witches."},
            {"role": "assistant", "content": "The three weird sisters appear to Macbeth on the heath. They speak in riddles and prophecy: all hail Macbeth, that shalt be king hereafter. Their words plant the seed of murder in his mind."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is jealousy?"},
            {"role": "assistant", "content": "It is the green-eyed monster which doth mock the meat it feeds on. Jealousy destroys from within. It needs no proof, for it creates its own evidence from nothing."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Who is Lady Macbeth?"},
            {"role": "assistant", "content": "Lady Macbeth is the wife of Macbeth. She drives him to murder the king, calling him a coward. But guilt consumes her and she walks in her sleep, washing imaginary blood from her hands."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Write about the forest."},
            {"role": "assistant", "content": "In the forest of Arden, the banished duke lives free. The trees are his court, the brook his music, and the deer his companions. Here, away from the lies of court, truth can be spoken."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "What is ambition?"},
            {"role": "assistant", "content": "Ambition is the soldier's virtue and the tyrant's vice. It builds kingdoms and destroys souls. Macbeth was ambitious, and it cost him everything. Caesar was ambitious, and they killed him for it."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Tell me about Ariel."},
            {"role": "assistant", "content": "Ariel is a spirit of air and fire, bound to serve Prospero. He can fly, become invisible, raise storms, and sing songs that enchant. He longs for freedom and Prospero promises to release him."},
        ]
    },
    # --- Edge cases: cortas, ambiguas, fuera de dominio ---
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Good morrow, friend. How may I serve you? Ask me of plays, of kings, of love, or of the human heart."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "Thank you."},
            {"role": "assistant", "content": "You are most welcome. If you have more questions about the plays, I am here to answer."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": "I don't understand."},
            {"role": "assistant", "content": "Fear not. Let me explain more simply. Which play or character would you like to know about?"},
        ]
    },
]


# =====================================================================
# PARTE 3: SFT TRAINING
# =====================================================================
# El SFT es conceptualmente simple:
#   1. Formatear cada ejemplo con el chat template
#   2. El modelo predice el siguiente token (igual que pre-training)
#   3. PERO: solo calculamos loss en los tokens del ASSISTANT
#      (no queremos que el modelo "aprenda a ser el usuario")
#
# En LLMs reales se usa "loss masking":
#   - Tokens de system/user: loss = 0 (se ignoran)
#   - Tokens de assistant: loss = cross_entropy (se aprenden)
#
# Esto ensena al modelo: "cuando veas <|user|>...<|end|><|assistant|>,
# genera una respuesta en el estilo de los ejemplos."
# =====================================================================


def create_sft_batch(dataset, tokenizer, max_seq_len=128):
    """
    Convierte el dataset SFT en tensores de training.

    Para cada ejemplo:
      1. Tokeniza con chat template
      2. Crea input_ids y labels
      3. Labels = -100 para tokens de system/user (loss ignorado)
      4. Labels = token_id para tokens de assistant (loss calculado)

    El -100 es la convencion de PyTorch: F.cross_entropy ignora -100.
    """
    all_input_ids = []
    all_labels = []

    for example in dataset:
        messages = example["messages"]
        token_ids = tokenizer.encode_chat(messages)

        # Truncar si es muy largo
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]

        # Crear labels con mascara
        # -100 = ignorar (system/user), token_id = aprender (assistant)
        labels = [-100] * len(token_ids)

        # Encontrar las regiones de assistant
        in_assistant = False
        for i, tid in enumerate(token_ids):
            if tid == tokenizer.assistant_id:
                in_assistant = True
                continue  # El token <|assistant|> mismo no se predice
            elif tid == tokenizer.end_id and in_assistant:
                labels[i] = tid  # El <|end|> del assistant SI se predice
                in_assistant = False
                continue
            elif tid in (tokenizer.system_id, tokenizer.user_id):
                in_assistant = False
                continue

            if in_assistant:
                labels[i] = tid

        # Shift: input = [0..n-1], labels = [1..n]
        # (el modelo predice el SIGUIENTE token)
        input_ids = token_ids[:-1]
        labels = labels[1:]

        # Padding a max_seq_len - 1
        pad_len = (max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return (
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
    )


def train_sft(model, tokenizer, dataset, epochs=30, lr=3e-4,
              max_seq_len=128, device='cpu'):
    """
    Supervised Fine-Tuning del modelo pre-entrenado.

    Detalles importantes:
    - Learning rate mas bajo que pre-training (3e-4 vs 1e-3)
      Para no destruir lo que ya aprendio.
    - Solo calculamos loss en tokens de assistant (loss masking)
    - Usamos el dataset completo como un solo batch (es pequeno)
    """
    model.train()
    model.to(device)

    # Crear batch
    input_ids, labels = create_sft_batch(dataset, tokenizer, max_seq_len)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Contar tokens de training reales (no masked)
    n_train_tokens = (labels != -100).sum().item()
    n_total_tokens = labels.numel()
    print(f"\n  Dataset SFT: {len(dataset)} ejemplos")
    print(f"  Tokens totales: {n_total_tokens:,}")
    print(f"  Tokens de training (assistant): {n_train_tokens:,} ({n_train_tokens/n_total_tokens*100:.1f}%)")
    print(f"  Tokens masked (system/user/pad): {n_total_tokens - n_train_tokens:,}")
    print(f"  Secuencia maxima: {max_seq_len}")
    print(f"  Learning rate: {lr}")
    print(f"  Epocas: {epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    print(f"\n  {'Epoca':>6} | {'Loss':>8} | {'LR':>10} | {'Tiempo':>8}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*8}")

    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        # Forward
        logits, _ = model(input_ids)  # [B, T, V], caches (ignored)

        # Flatten para cross_entropy
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=-100,  # Ignora tokens masked
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        dt = time.time() - t_epoch
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6} | {loss.item():>8.4f} | {current_lr:>10.6f} | {dt:>7.1f}s")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = copy.deepcopy(model.state_dict())

    total_time = time.time() - t0
    print(f"\n  Mejor loss: {best_loss:.4f}")
    print(f"  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Restaurar mejor modelo
    model.load_state_dict(best_state)
    return best_loss


# =====================================================================
# PARTE 4: GENERACION CON CHAT TEMPLATE
# =====================================================================

@torch.no_grad()
def generate_chat(model, tokenizer, messages, max_new_tokens=100,
                  temperature=0.7, top_k=40, device='cpu'):
    """
    Genera una respuesta dado un historial de mensajes.

    Proceso:
    1. Codifica los mensajes con el chat template
    2. Agrega <|assistant|> al final (el modelo debe completar)
    3. Genera token por token hasta <|end|> o max_tokens
    4. Decodifica solo la parte generada

    Esto es EXACTAMENTE lo que hace la API de OpenAI/Anthropic
    cuando le mandas un array de mensajes.
    """
    model.eval()
    model.to(device)

    # Agregar un mensaje vacio de assistant para que el modelo complete
    gen_messages = messages + [{"role": "assistant", "content": ""}]
    input_ids = tokenizer.encode_chat_for_generation(gen_messages)

    # Tensor
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = ids.shape[1]

    generated = []
    max_total_len = 128  # Model's max sequence length (RoPE limit)

    for _ in range(max_new_tokens):
        # Truncar si excedemos el limite del modelo
        if ids.shape[1] >= max_total_len:
            break

        # Forward (sin KV-cache por simplicidad, como Ej1/Ej2)
        logits, _ = model(ids)  # [1, T, V], caches (ignored)
        next_logits = logits[0, -1, :]  # Ultimo token

        # Temperatura
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k
        if top_k > 0:
            values, indices = torch.topk(next_logits, top_k)
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(-1, indices, values)
            next_logits = mask

        # Softmax + sampling
        probs = F.softmax(next_logits, dim=-1)

        if temperature > 0:
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(probs).unsqueeze(0)

        next_id = next_token.item()

        # Parar si genera <|end|>
        if next_id == tokenizer.end_id:
            break

        generated.append(next_id)
        ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

    # Decodificar solo la parte generada
    return tokenizer.decode(generated)


# =====================================================================
# PARTE 5: MAIN -- DEMO COMPLETA
# =====================================================================

def main():
    print("=" * 70)
    print("EJERCICIO 3: De Transformer a Mini-LLM (Supervised Fine-Tuning)")
    print("=" * 70)
    print()
    print("Que hacemos en este ejercicio:")
    print("  1. Cargamos el Transformer pre-entrenado del Ej2 (Shakespeare)")
    print("  2. Le agregamos un chat template (<|user|>, <|assistant|>, etc.)")
    print("  3. Lo fine-tuneamos con 33 ejemplos de instruccion-respuesta")
    print("  4. Comparamos: modelo BASE vs modelo SFT")
    print()
    print("La leccion: SFT no agrega 'conocimiento'. Agrega FORMATO.")
    print("El modelo aprende CUANDO y COMO responder, no QUE saber.")
    print()

    # ------------------------------------------------------------------
    # Paso 1: Cargar modelo pre-entrenado del Ej2
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PASO 1: Cargar modelo pre-entrenado del Ej2")
    print("=" * 70)

    ej2_dir = os.path.join(os.path.dirname(__file__), '..', 'ej2_training')
    model_path = os.path.join(ej2_dir, 'best_model_v2.pt')
    tokenizer_path = os.path.join(ej2_dir, 'tokenizer_v2.json')

    if not os.path.exists(model_path):
        print(f"\n  ERROR: No se encuentra {model_path}")
        print(f"  Debes ejecutar primero el Ejercicio 2 (train_real.py)")
        sys.exit(1)

    # Configuracion del modelo (DEBE coincidir con Ej2 -- ver train_real.py)
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 384
    MAX_SEQ = 128
    DROPOUT = 0.1

    # Cargar tokenizer base
    chat_tokenizer = ChatTokenizer(tokenizer_path)
    print(f"\n  Tokenizer base: {chat_tokenizer.base_vocab_size} tokens")
    print(f"  Tokens especiales: {list(chat_tokenizer.special_tokens.keys())}")
    print(f"  Vocabulario total: {chat_tokenizer.vocab_size} tokens")

    # Cargar pesos pre-entrenados del Ej2
    checkpoint_raw = torch.load(model_path, map_location='cpu', weights_only=False)
    # El checkpoint del Ej2 es un dict con 'model_state', 'val_loss', 'epoch', 'config'
    if isinstance(checkpoint_raw, dict) and 'model_state' in checkpoint_raw:
        checkpoint = checkpoint_raw['model_state']
        print(f"  Checkpoint Ej2: epoch {checkpoint_raw.get('epoch', '?')}, val_loss {checkpoint_raw.get('val_loss', '?'):.4f}")
    else:
        checkpoint = checkpoint_raw

    # Crear modelo con vocabulario EXPANDIDO (base + tokens especiales)
    # El embedding y output head tienen 5 filas mas que el original.
    model = TransformerLMv2(
        vocab_size=chat_tokenizer.vocab_size,  # 1000 + 5 = 1005
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        dropout=DROPOUT,
    )

    # Cargar pesos pre-entrenados (solo los que coinciden en tamaño)
    # tok_emb y head tienen tamaño diferente (1000 vs 1005), se cargan parcialmente
    pretrained_state = checkpoint
    model_state = model.state_dict()

    loaded = 0
    skipped = []
    for key in pretrained_state:
        if key in model_state and pretrained_state[key].shape == model_state[key].shape:
            model_state[key] = pretrained_state[key]
            loaded += 1
        else:
            skipped.append(key)

    model.load_state_dict(model_state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Modelo cargado: {n_params:,} parametros")
    print(f"  Pesos pre-entrenados: {loaded} cargados, {len(skipped)} con tamaño diferente")
    if skipped:
        print(f"  Tamaño diferente (esperado -- vocab expandido): {skipped}")

    # Copiar pesos del embedding base (1000 tokens) al modelo expandido (1005)
    # Para los 5 tokens especiales, inicializamos con la media de los embeddings
    # existentes. Esto es lo que hacen los LLMs reales al agregar tokens.
    with torch.no_grad():
        # Copiar las primeras 1000 filas del embedding pre-entrenado
        if 'tok_emb.weight' in pretrained_state:
            pre_emb = pretrained_state['tok_emb.weight']
            model.tok_emb.weight[:pre_emb.shape[0]] = pre_emb

            # Inicializar tokens especiales con la media
            base_mean = pre_emb.mean(dim=0)
            for i in range(len(SPECIAL_TOKENS)):
                model.tok_emb.weight[chat_tokenizer.base_vocab_size + i] = base_mean

        # head.weight esta tied a tok_emb.weight, asi que ya se actualizo
        # (model.head.weight IS model.tok_emb.weight por weight tying)

    print(f"  Embeddings de tokens especiales inicializados con media de los existentes")
    print(f"  Weight tying: head.weight es tok_emb.weight = {model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()}")

    # ------------------------------------------------------------------
    # Paso 2: Modelo BASE -- ver como responde SIN SFT
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PASO 2: Modelo BASE (sin SFT) -- intenta 'chatear'")
    print("=" * 70)
    print()
    print("  El modelo base fue entrenado para COMPLETAR Shakespeare.")
    print("  Si le ponemos un chat template, no sabe que hacer con el.")
    print("  Va a ignorar la estructura y generar texto libre.")
    print()

    test_prompts = [
        [{"role": "user", "content": "Who is Romeo?"}],
        [{"role": "user", "content": "Tell me about the night."}],
        [{"role": "user", "content": "What is love?"}],
    ]

    for messages in test_prompts:
        user_msg = messages[0]["content"]
        response = generate_chat(
            model, chat_tokenizer, messages,
            max_new_tokens=60, temperature=0.7
        )
        print(f"  Pregunta: \"{user_msg}\"")
        print(f"  Base:     \"{response[:200]}\"")
        print()

    # Guardar estado base para comparacion
    base_state = copy.deepcopy(model.state_dict())

    # ------------------------------------------------------------------
    # Paso 3: SFT -- Fine-tuning supervisado
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PASO 3: Supervised Fine-Tuning (SFT)")
    print("=" * 70)
    print()
    print("  Ahora entrenamos el modelo con el chat template.")
    print("  Solo calculamos loss en los tokens del ASSISTANT.")
    print("  El modelo aprende: 'despues de <|user|>...<|end|><|assistant|>,")
    print("  debo generar una respuesta relevante.'")

    # Expandir dataset: multi-turn genera mas de un training example
    expanded_dataset = []
    for example in SFT_DATASET:
        msgs = example["messages"]
        # Para multi-turn, crear un ejemplo por cada respuesta de assistant
        assistant_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for ai in assistant_indices:
            # Contexto hasta este assistant (inclusive)
            context = msgs[:ai+1]
            expanded_dataset.append({"messages": context})

    print(f"  Dataset original: {len(SFT_DATASET)} conversaciones")
    print(f"  Dataset expandido: {len(expanded_dataset)} ejemplos de training")

    best_loss = train_sft(
        model, chat_tokenizer, expanded_dataset,
        epochs=50, lr=5e-4, max_seq_len=128
    )

    # Guardar estado SFT ANTES de la comparacion
    sft_state = copy.deepcopy(model.state_dict())
    save_path = os.path.join(os.path.dirname(__file__), 'sft_model.pt')
    torch.save(sft_state, save_path)
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  Modelo SFT guardado: {save_path} ({model_size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Paso 4: Comparacion -- BASE vs SFT
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PASO 4: Comparacion -- BASE vs SFT")
    print("=" * 70)
    print()
    print("  Mismas preguntas, dos modelos diferentes.")
    print("  Observa la diferencia en FORMATO (no en 'inteligencia').")
    print()

    # Preguntas de test (incluye preguntas NO vistas en training)
    comparison_prompts = [
        # Vistas en training
        ("Who is Romeo?", True),
        ("What is love?", True),
        ("Tell me about Macbeth.", True),
        # NO vistas en training (generalizacion)
        ("Who is the bravest?", False),
        ("Speak of the sea.", False),
        ("What makes a man noble?", False),
    ]

    for user_msg, in_training in comparison_prompts:
        messages = [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": user_msg},
        ]

        # Generar con modelo SFT (estado actual)
        model.load_state_dict(sft_state)
        sft_response = generate_chat(
            model, chat_tokenizer, messages,
            max_new_tokens=80, temperature=0.5
        )

        # Generar con modelo BASE
        model.load_state_dict(base_state)
        base_response = generate_chat(
            model, chat_tokenizer, messages,
            max_new_tokens=80, temperature=0.5
        )

        tag = "[VISTO]" if in_training else "[NUEVO]"
        print(f"  {tag} Pregunta: \"{user_msg}\"")
        print(f"    BASE: \"{base_response[:200]}\"")
        print(f"    SFT:  \"{sft_response[:200]}\"")
        print()

    # Restaurar SFT para las pruebas finales
    model.load_state_dict(sft_state)

    # ------------------------------------------------------------------
    # Paso 6: Demo interactiva final
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PASO 5: Demo interactiva -- El mini-LLM responde")
    print("=" * 70)
    print()

    final_prompts = [
        "Hello.",
        "Who is Hamlet?",
        "Write a short verse.",
        "Compare Romeo and Hamlet.",
        "What is honour?",
        "Tell me about a forest.",       # No exactamente en training
        "Why do stories matter?",         # Variante de una pregunta vista
    ]

    for user_msg in final_prompts:
        messages = [
            {"role": "system", "content": "You are a Shakespeare scholar."},
            {"role": "user", "content": user_msg},
        ]
        response = generate_chat(
            model, chat_tokenizer, messages,
            max_new_tokens=80, temperature=0.5
        )
        print(f"  User: \"{user_msg}\"")
        print(f"  LLM:  \"{response[:300]}\"")
        print()

    # ------------------------------------------------------------------
    # Explicacion final
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("LECCION DEL EJERCICIO 3")
    print("=" * 70)
    print("""
Lo que demostramos:

  1. PRE-TRAINING vs SFT:
     - Pre-training (Ej2): el modelo aprende PATRONES del texto
     - SFT (Ej3): el modelo aprende un FORMATO de respuesta
     - El SFT no agrega conocimiento -- solo cambia el comportamiento

  2. CHAT TEMPLATE:
     Los tokens <|user|>, <|assistant|>, <|end|> son CONVENCIONES.
     No son magicos. El modelo aprende que despues de <|assistant|>
     debe generar texto que "responda" al <|user|>.

  3. LOSS MASKING:
     Solo entrenamos en tokens de assistant.
     Esto es critico: si entrenamos en TODO (incluido el user),
     el modelo aprenderia a "ser el usuario" tambien.

  4. GENERALIZACION LIMITADA:
     Con 33 ejemplos de SFT, el modelo generaliza un POCO.
     Los LLMs reales usan 100K+ ejemplos de SFT.
     Pero el principio es identico.

  5. LA ANALOGIA CON LA SEGURIDAD:
     Un modelo SFT que "rechaza" preguntas peligrosas
     solo aprendio un PATRON estadistico de rechazo.
     No "entiende" por que es peligroso.
     Por eso los jailbreaks funcionan: cambian el contexto
     para que el patron de rechazo no se active.

En el siguiente ejercicio construiremos la INFERENCIA real:
KV-cache, sampling strategies, y generacion token-a-token
con visualizacion de probabilidades.
""")

    print(f"EXIT CODE: 0")


if __name__ == "__main__":
    main()
