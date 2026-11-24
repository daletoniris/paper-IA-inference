
## La Verdad sobre los Agentes de IA

**Autor:** Daniel Dieser -- Robotics, AI en Security information Research, 2026

---

## Que es esto?

Un paper tecnico con codigo funcional que explica, desde cero, como
funcionan los agentes de IA y por que la seguridad no se resuelve con
un system prompt.

La progresion construye cada concepto sobre el anterior:

```
Neurona -> Red Neuronal -> Transformer -> LLM -> Inferencia -> Tool Calling -> Agente -> Seguridad
```

Al terminar, el lector entiende exactamente que hay detras de un
"agente de IA" y puede construir uno desde cero.

---

## Estructura del repositorio

```
paper-IA-inference/
|
|-- README.md                           # Este archivo
|-- requirements.txt                    # Dependencias Python (NumPy, PyTorch, OpenAI SDK)
|
|-- paper/                              # El paper completo (8 partes en Markdown)
|   |-- 00-indice.md                    # Indice general
|   |-- 01-la-neurona.md                # Perceptron, red neuronal, backpropagation
|   |-- 02-el-transformer.md            # Attention, multi-head, positional encoding
|   |-- 03-del-transformer-al-llm.md    # Escala, SFT, RLHF, inferencia, sampling
|   |-- 04-de-modelo-a-agente.md        # Runtime, tool calling, loop, memoria
|   |-- 05-minimalista-vs-avanzado.md   # Mismo modelo, dos runtimes, resultados distintos
|   |-- 06-modelo-de-seguridad.md       # 5 capas de defensa (la parte critica)
|   |-- 07-conclusion.md                # Sintesis y reflexiones
|   |-- 08-ataque-real.md               # 6 escenarios de ataque con y sin defensas
|
|-- code/                               # Ejemplos rapidos del paper (minutos)
|   |-- perceptron.py                   # Perceptron desde cero (0 dependencias)
|   |-- red_neuronal.py                 # Red neuronal XOR (NumPy, 17 parametros)
|   |-- mini_transformer.py            # Transformer decoder-only (PyTorch, 110K params)
|   |-- agente_minimalista.py          # Agente con 3 tools (OpenAI SDK + Ollama)
|   |-- agente_seguro.py              # Agente con 5 capas de defensa (demo)
|   |-- demo_ataque.py                 # 6 ataques: con y sin defensas (no requiere LLM)
|
|-- ejercicios/                         # 6 ejercicios profundos (el corazon del repo)
|   |-- README.md                       # Guia de los ejercicios
|   |-- ej1_transformer/               # Transformer REAL desde cero (BPE, RoPE, SwiGLU)
|   |-- ej2_training/                  # Entrenar con Shakespeare (transfer learning)
|   |-- ej3_mini_llm/                  # SFT: convertir transformer en chatbot
|   |-- ej4_inference/                 # KV-cache, temperature, top-p, top-k
|   |-- ej5_tool_calling/             # Runtime de tool calling desde cero
|   |-- ej6_agente_completo/          # Agente completo con todas las piezas
|
|-- agente-seguro/                      # Agente con 5 capas de seguridad (produccion)
|   |-- README.md                       # Documentacion completa del agente
|   |-- agent.py                       # Loop principal (OpenRouter + Ollama)
|   |-- tools.py                       # 5 herramientas con validacion estricta
|   |-- memory.py                      # Memoria persistente en JSON
|   |-- .env.example                   # Configuracion de ejemplo (copiar a .env)
|   |-- requirements.txt              # Dependencias (openai, python-dotenv)
|
|-- agente-avanzado/                    # Agente sin restricciones (experimentacion)
|   |-- README.md                       # Explicacion + link a TokioAI
|   |-- ejemplo_sin_restricciones.py   # Demo: agente que ejecuta comandos reales
```

---

## Inicio rapido

### 1. Clonar el repositorio

```bash
git clone https://github.com/daletoniris/paper-IA-inference.git
cd paper-IA-inference
```

### 2. Instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Ejecutar ejemplos rapidos (sin LLM)

```bash
# Perceptron desde cero (0 dependencias, 2 segundos)
python3 code/perceptron.py

# Red neuronal XOR (NumPy, 2 segundos)
python3 code/red_neuronal.py

# Mini-Transformer (PyTorch, 30 segundos)
python3 code/mini_transformer.py

# Demo de ataque (sin LLM, muestra las 5 capas de defensa)
python3 code/demo_ataque.py
```

### 4. Ejercicios profundos (requieren PyTorch)

```bash
# Transformer REAL desde cero (~5 min en Pi, ~1 min en laptop)
python3 ejercicios/ej1_transformer/transformer_real.py

# Entrenamiento con Shakespeare (~8 min)
python3 ejercicios/ej2_training/train_real.py

# SFT: de transformer a chatbot (~8 min)
python3 ejercicios/ej3_mini_llm/mini_llm.py

# Inferencia: KV-cache + sampling (~1 min)
python3 ejercicios/ej4_inference/inference_real.py
```

### 5. Tool Calling y Agente (requieren Ollama)

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo (una vez, ~2.5 GB)
ollama pull qwen3:4b

# Iniciar Ollama en otra terminal
ollama serve

# Tool calling desde cero
python3 ejercicios/ej5_tool_calling/tool_calling.py

# Agente completo
python3 ejercicios/ej6_agente_completo/agente.py
```

### 6. Agente seguro con interfaz interactiva

```bash
cd agente-seguro
cp .env.example .env
# Editar .env con tu API key (OpenRouter) o config de Ollama
pip install -r requirements.txt
python3 agent.py
```

### 7. Agente avanzado (sin restricciones)

```bash
# Requiere Ollama corriendo
python3 agente-avanzado/ejemplo_sin_restricciones.py

# Para el agente completo avanzado (TokioAI):
# https://github.com/TokioAI/tokioai
```

---

## Requisitos

### Hardware minimo
- **CPU**: cualquier x86_64 o ARM64 (probado en Raspberry Pi 5)
- **RAM**: 8 GB minimo (para Qwen3:4b via Ollama)
- **Disco**: 5 GB libres
- **GPU**: NO necesaria (todo corre en CPU)

### Software
- Python 3.10 o superior
- pip
- Git
- Ollama (solo para ejercicios 5-6 y el agente)

---

## Sobre el modelo

El paper usa **Qwen 3 4B** (`qwen3:4b` en Ollama) porque:

1. Soporta **tool calling** nativo (no todos los modelos lo hacen bien)
2. Corre en CPU con 8 GB de RAM (no necesita GPU)
3. Es **open-source y gratuito** (Apache 2.0)
4. Es suficientemente capaz para demostrar todos los conceptos
5. Se puede descargar con un solo comando (`ollama pull qwen3:4b`)

---

## El paper resumido en una frase

> Un modelo de IA no "hackea". Es un generador de texto estadístico o tecnicamente llamado LLM entrenado con la arquitectura Transformer.
> Si tu agente hace cosas peligrosas, el problema esta en tu RUNTIME,
> no en el modelo. La discusión de que los modelos de AI se deberían haber entrenado con otro tipo de información se debería haber dado antes de "CREARLOS", en términos de acceso a que info o no deberíamos predecir con un Modelo, ahora tenemos que enfocar la industria y la seguridad en función del ADN de los LLMS. Antropomorfizar la IA puede ser un gran error.

---

## Licencia

Uso educativo. El paper y el codigo son de uso libre para aprendizaje
e investigacion.
