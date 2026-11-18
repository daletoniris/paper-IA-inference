# Ejercicios Practicos

Estos 6 ejercicios acompanian el paper **"Hackers Artificiales o Sistemas Mal Disenados?"**
y cubren la progresion completa: de una neurona individual a un agente de IA funcional.

Cada ejercicio:
- Es un archivo Python autocontenido (se ejecuta directo)
- Incluye un `output.log` con la salida esperada
- Esta extensamente comentado (el codigo ES la documentacion)
- Corre en CPU en una Raspberry Pi 5 o cualquier laptop

---

## Progresion

```
Ej1: Transformer   -> Construis la arquitectura desde cero (BPE, RoPE, SwiGLU, KV-cache)
Ej2: Training      -> Entrenas con texto real (Shakespeare) y ves como aprende
Ej3: Mini-LLM      -> Haces SFT para convertir el modelo en un chatbot
Ej4: Inferencia    -> Implementas sampling (temperature, top-p, top-k) y ves probabilidades
Ej5: Tool Calling  -> Construis el runtime que convierte texto en acciones
Ej6: Agente        -> Combinas todo: modelo + runtime + herramientas + memoria = agente
```

---

## Como ejecutar

### Requisitos
```bash
pip install torch numpy    # Para ej1-ej4
pip install openai         # Para ej5-ej6 (solo si usas modelo real)
```

### Ejercicios 1-4 (no requieren LLM externo)
```bash
python3 ejercicios/ej1_transformer/transformer_real.py    # ~5 min en Pi, ~1 min en laptop
python3 ejercicios/ej2_training/train_real.py             # ~8 min en Pi, ~2 min en laptop
python3 ejercicios/ej3_mini_llm/mini_llm.py               # ~8 min en Pi, ~2 min en laptop
python3 ejercicios/ej4_inference/inference_real.py         # ~1 min (no entrena, solo infiere)
```

### Ejercicios 5-6 (requieren Ollama con qwen3:4b)
```bash
# Primero: instalar y correr Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama serve   # Dejar corriendo en otra terminal

# Luego ejecutar:
python3 ejercicios/ej5_tool_calling/tool_calling.py       # ~2 min
python3 ejercicios/ej6_agente_completo/agente.py           # ~2 min
```

> **Nota:** Los ejercicios 5 y 6 incluyen demos simuladas que corren SIN modelo.
> Si tenes Ollama corriendo, tambien ejecutan la parte interactiva con el modelo real.

---

## Archivos en cada ejercicio

| Archivo            | Que es                                                    |
|--------------------|-----------------------------------------------------------|
| `*.py`             | El script principal -- ejecutalo con `python3`            |
| `output.log`       | Salida esperada -- compara con tu ejecucion               |
| `best_model*.pt`   | Modelo entrenado (prueba de que funciona)                 |
| `tokenizer*.json`  | Vocabulario BPE construido durante training               |
| `data_tinystory.txt` | Corpus de entrenamiento (Shakespeare, dominio publico)  |

---

## Si algo falla

1. **"ModuleNotFoundError: No module named 'torch'"**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Ejercicio 5-6 dice "Ollama no encontrado"**
   ```bash
   # Verificar que Ollama esta corriendo:
   curl http://localhost:11434/api/tags
   # Si no responde, inicialo: ollama serve
   ```

3. **Ejercicio tarda mucho**
   - En Raspberry Pi 5, ej1-ej3 tardan 5-8 minutos cada uno
   - En laptop moderna, 1-2 minutos
   - Si tarda mas de 20 minutos, algo esta mal

4. **"CUDA not available"**
   - Esto esta bien -- todos los ejercicios corren en CPU
   - Si tenes GPU, PyTorch la usara automaticamente y sera mas rapido
