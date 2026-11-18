# Agente Seguro -- IA con Tool Calling y 5 Capas de Defensa

Un agente de inteligencia artificial que usa **tool calling** para interactuar
con el mundo real: hacer calculos, analizar codigo, detectar vulnerabilidades,
generar scripts, y mas.

Implementa las **5 capas de seguridad** del paper:
1. **Validacion de inputs** -- detecta prompt injection antes de llegar al modelo
2. **Registro de herramientas** -- solo ejecuta funciones del TOOL_REGISTRY
3. **Sanitizacion de outputs** -- valida los resultados antes de devolverlos
4. **Segmentacion de permisos** -- cada herramienta tiene permisos acotados
5. **Auditoria completa** -- log de cada accion para analisis post-mortem

Soporta **dos backends** intercambiables:
- **OpenRouter** (API en la nube) -- rapido, modelos potentes, necesita API key
- **Ollama** (local o VPS) -- privado, sin costo de API, funciona offline

---

## Instalacion paso a paso

### 1. Preparar entorno Python

```bash
cd agente-seguro/
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configurar el backend

```bash
cp .env.example .env
```

Edita `.env` y elige una opcion:

#### Opcion A: OpenRouter (recomendado para empezar)

1. Crea una cuenta gratis en [openrouter.ai](https://openrouter.ai)
2. Genera una API key en [openrouter.ai/keys](https://openrouter.ai/keys)
3. Edita `.env`:

```env
BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-v1-TU_API_KEY_AQUI
OPENROUTER_MODEL=google/gemini-2.5-flash
```

#### Opcion B: Ollama (local, gratis)

1. Instala Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Descarga un modelo: `ollama pull qwen3:4b`
3. Inicia Ollama: `ollama serve` (en otra terminal)
4. Edita `.env`:

```env
BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
```

### 3. Ejecutar

```bash
python3 agent.py
```

O usar el script de inicio:
```bash
./run.sh      # Linux/macOS
run.bat       # Windows
```

---

## Uso

```
Tu > cuanto es 2^16 + 512?
  [TOOL] calculator({'expression': '2**16 + 512'})
    -> {'expresion': '2**16 + 512', 'resultado': 66048}
  [2s]

AI > 2^16 + 512 = 66,048.
```

### Herramientas disponibles

| Herramienta      | Que hace                                              |
|------------------|-------------------------------------------------------|
| `system_status`  | Info del SO, disco, Python, hora UTC                  |
| `calculator`     | Calcula expresiones aritmeticas (sin eval, seguro)    |
| `write_script`   | Genera scripts Python (guarda, NUNCA ejecuta)         |
| `security_scan`  | Analiza codigo buscando vulnerabilidades (5 lenguajes)|
| `count_lines`    | Cuenta lineas de codigo, comentarios y vacias         |

### Comandos especiales

| Comando     | Que hace                                     |
|-------------|----------------------------------------------|
| `/ayuda`    | Muestra la ayuda                             |
| `/limpiar`  | Borra el historial de conversacion           |
| `/backend`  | Cambia entre OpenRouter y Ollama en caliente |
| `/salir`    | Termina el agente                            |

---

## Arquitectura

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |   OpenRouter API |
|   Tu (humano)    |---->|   agent.py       |---->|   o Ollama local |
|                  |     |   (orquestador)  |     +------------------+
|                  |<----|                  |
+------------------+     +--------+---------+
                                  |
                                  | tool call?
                                  v
                         +------------------+
                         |   tools.py       |
                         |   5 herramientas |
                         |   con validacion |
                         +------------------+
                                  |
                                  v
                         +------------------+
                         |   memory.py      |
                         |   persistencia   |
                         +------------------+
```

### Flujo:

1. Escribis un mensaje
2. `agent.py` envia tu mensaje + historial al modelo (OpenRouter o Ollama)
3. El modelo responde con texto directo O con un tool call
4. Si es tool call, `agent.py` ejecuta la herramienta via `tools.py`
5. El resultado se le devuelve al modelo para la respuesta final
6. El historial se guarda en `memory.json`

### Archivos

| Archivo            | Que hace                                          |
|--------------------|---------------------------------------------------|
| `agent.py`         | Loop principal: recibe input, llama al modelo, ejecuta tools |
| `tools.py`         | Las 5 herramientas + toda la validacion de seguridad |
| `memory.py`        | Lee/escribe historial en `memory.json`             |
| `.env`             | Tu configuracion (backend, API key, modelo)        |
| `.env.example`     | Plantilla de configuracion (copiar a `.env`)       |
| `requirements.txt` | Dependencias Python (openai, python-dotenv)        |
| `run.sh` / `run.bat` | Scripts de inicio rapido                        |

---

## Velocidades tipicas

| Dispositivo          | Modelo     | Velocidad     | Tiempo por respuesta |
|----------------------|------------|---------------|----------------------|
| Raspberry Pi 5       | qwen3:4b   | ~2-3 tok/s    | 60-180s              |
| Laptop (CPU)         | qwen3:4b   | ~15-25 tok/s  | 5-10s                |
| PC con GPU           | qwen3:4b   | ~40-60 tok/s  | 2-5s                 |
| OpenRouter (Gemini)  | gemini-flash | ~100 tok/s  | 1-3s                 |

---

## Si algo falla

### "Ollama no esta corriendo"
```bash
# Verificar:
curl http://localhost:11434/api/tags

# Si no responde, iniciar:
ollama serve
```

### "Modelo no encontrado"
```bash
# Descargar el modelo:
ollama pull qwen3:4b
```

### "OPENROUTER_API_KEY invalida"
1. Verifica que copiaste bien la key en `.env`
2. La key debe empezar con `sk-or-v1-`
3. Revisa que tengas credito en [openrouter.ai/credits](https://openrouter.ai/credits)

### El modelo no usa herramientas (responde texto directo)
- Algunos modelos son malos para tool calling
- Modelos probados que funcionan bien: `google/gemini-2.5-flash`, `qwen3:4b`, `qwen3:8b`
- Modelos que NO funcionan bien: `llama3:8b`, `phi3` (no soportan tool calling nativo)
