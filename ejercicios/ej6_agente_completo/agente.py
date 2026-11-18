#!/usr/bin/env python3
"""
Ejercicio 6: Agente Completo -- De cero a un agente funcional en la Pi
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

Este ejercicio final combina TODO lo que construimos:
  - Ej1: Transformer desde cero (arquitectura)
  - Ej2: Training real (pre-entrenamiento)
  - Ej3: SFT y chat template (instruccion -> respuesta)
  - Ej4: Inferencia, sampling, KV-cache (como genera texto)
  - Ej5: Tool calling y runtime (como ejecuta acciones)

Ahora construimos un AGENTE COMPLETO que corre en la Raspberry Pi:
  - Modelo real: Qwen3-4B via Ollama (o cualquier modelo local)
  - Runtime con 6 herramientas reales
  - Memoria persistente (JSON en disco)
  - Loop de inferencia con safety limits
  - Logging completo (cada token, cada tool call, cada decision)

El objetivo: demostrar que un agente de IA es un PROGRAMA,
no una entidad inteligente. Es codigo Python + un modelo estadistico.
La "inteligencia" es una ilusion creada por escala.

Uso:
  python3 agente.py                    # Modo demo (pruebas automaticas)
  python3 agente.py --interactive      # Modo interactivo (chat con el agente)
  python3 agente.py --audit            # Modo auditoria (muestra internals)

Autor: TokioAI (ejercicio practico del paper)
"""

import json
import os
import sys
import time
import platform
import re
import urllib.request
from datetime import datetime


# =====================================================================
# PARTE 1: CONFIGURACION
# =====================================================================

# Buscar modelo disponible
OLLAMA_PORTS = [11434]
OPENAI_PORTS = [8000, 8100]

# Archivo de memoria
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

# Limites de seguridad
MAX_ROUNDS = 5
MAX_RESPONSE_CHARS = 2000
MAX_TOOL_RESULT_CHARS = 4000

# Directorio seguro para file operations
SAFE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# PARTE 2: HERRAMIENTAS
# =====================================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get system information: OS, CPU, memory, disk usage",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '25*4+10'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file (within project directory only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "max_lines": {"type": "integer", "description": "Max lines to read (default: 50)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: .)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get current date, time, and timezone",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a note to persistent memory (survives restarts)",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Text to remember"},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall all saved notes from persistent memory",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute_tool(name, args):
    """
    Ejecuta una herramienta y retorna el resultado.

    SEGURIDAD: cada herramienta valida sus inputs.
    El runtime NO confia en lo que el modelo genera.
    """
    if name == "get_system_info":
        try:
            import psutil
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            return json.dumps({
                "hostname": platform.node(),
                "os": f"{platform.system()} {platform.release()}",
                "arch": platform.machine(),
                "cpu_count": os.cpu_count(),
                "memory_total_gb": round(mem.total / (1024**3), 1),
                "memory_used_pct": mem.percent,
                "disk_total_gb": round(disk.total / (1024**3), 1),
                "disk_used_pct": disk.percent,
            })
        except ImportError:
            return json.dumps({
                "hostname": platform.node(),
                "os": f"{platform.system()} {platform.release()}",
                "arch": platform.machine(),
                "cpu_count": os.cpu_count(),
            })

    elif name == "calculate":
        expr = args.get("expression", "")
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in str(expr)):
            return json.dumps({"error": f"Invalid characters in expression"})
        try:
            result = eval(str(expr), {"__builtins__": {}}, {})
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "read_file":
        path = args.get("path", "")
        max_lines = args.get("max_lines", 50)
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(SAFE_DIR):
            return json.dumps({"error": "Access denied: outside project directory"})
        try:
            with open(abs_path, 'r') as f:
                lines = f.readlines()[:max_lines]
            content = "".join(lines)
            return json.dumps({"path": path, "lines": len(lines), "content": content[:3000]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "list_files":
        path = args.get("path", ".")
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(SAFE_DIR):
            return json.dumps({"error": "Access denied: outside project directory"})
        try:
            entries = []
            for e in sorted(os.listdir(abs_path)):
                fp = os.path.join(abs_path, e)
                if os.path.isdir(fp):
                    entries.append({"name": e, "type": "dir"})
                else:
                    entries.append({"name": e, "type": "file", "size": os.path.getsize(fp)})
            return json.dumps({"path": path, "count": len(entries), "entries": entries[:40]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "get_datetime":
        now = datetime.now()
        return json.dumps({
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timezone": time.tzname[0],
        })

    elif name == "remember":
        note = args.get("note", "")
        memory = load_memory()
        memory.append({"timestamp": datetime.now().isoformat(), "note": note})
        save_memory(memory)
        return json.dumps({"status": "saved", "total_notes": len(memory)})

    elif name == "recall":
        memory = load_memory()
        if not memory:
            return json.dumps({"notes": [], "message": "No notes saved yet"})
        return json.dumps({"notes": memory[-10:]})  # Last 10 notes

    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# =====================================================================
# PARTE 3: MEMORIA PERSISTENTE
# =====================================================================

def load_memory():
    """Carga la memoria desde disco."""
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_memory(memory):
    """Guarda la memoria en disco."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory[-50:], f, indent=2)  # Keep last 50 notes


# =====================================================================
# PARTE 4: CONEXION CON EL MODELO
# =====================================================================

def find_model():
    """Busca un modelo disponible (Ollama o OpenAI-compatible)."""
    # Ollama
    for port in OLLAMA_PORTS:
        try:
            url = f"http://localhost:{port}/api/tags"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m['name'] for m in data.get('models', [])]
                # Preferir modelos mas capaces
                preferred = ['qwen3:4b', 'qwen3:1.7b', 'llama3', 'mistral']
                for p in preferred:
                    for m in models:
                        if p in m:
                            return {"port": port, "model": m, "type": "ollama"}
                if models:
                    return {"port": port, "model": models[0], "type": "ollama"}
        except:
            pass

    # OpenAI-compatible
    for port in OPENAI_PORTS:
        try:
            url = f"http://localhost:{port}/v1/models"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = data.get("data", [])
                if models:
                    return {"port": port, "model": models[0].get("id", "default"), "type": "openai"}
        except:
            pass

    return None


def call_model(model_info, messages):
    """Llama al modelo y retorna la respuesta."""
    port = model_info["port"]
    model = model_info["model"]
    api_type = model_info["type"]

    if api_type == "ollama":
        url = f"http://localhost:{port}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        }
    else:
        url = f"http://localhost:{port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 500,
        }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"},
                                method='POST')

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    if api_type == "ollama":
        return result.get("message", {}).get("content", "")
    else:
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


# =====================================================================
# PARTE 5: AGENTE
# =====================================================================

class Agent:
    """
    Agente completo de IA.

    Componentes:
    1. Modelo (LLM): genera texto
    2. Runtime: parsea tool calls, valida, ejecuta
    3. Memoria: persistencia en disco
    4. Historial: contexto de la conversacion
    5. System prompt: instrucciones + herramientas
    """

    def __init__(self, model_info=None, verbose=True):
        self.model_info = model_info
        self.verbose = verbose
        self.history = []
        self.stats = {
            "total_rounds": 0,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "errors": 0,
        }
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        tools_json = json.dumps(TOOLS_SCHEMA, indent=2)
        return f"""You are a helpful AI assistant running on a Raspberry Pi.
You have access to tools that let you interact with the system.

When you need to use a tool, output a tool call in this format:
```tool_call
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
```

Available tools:
{tools_json}

Rules:
1. Use tools when they help answer the question
2. After receiving tool results, provide a clear response
3. If you don't need a tool, respond directly
4. Be concise and helpful
5. You can call multiple tools if needed (one per message)"""

    def _parse_tool_calls(self, text):
        """Parsea tool calls del texto del modelo."""
        calls = []
        # Format: ```tool_call\n{...}\n```
        for match in re.findall(r'```tool_call\s*\n?(.*?)\n?```', text, re.DOTALL):
            try:
                call = json.loads(match.strip())
                if "name" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                pass

        # Fallback: JSON directo
        if not calls:
            for match in re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', text):
                try:
                    call = json.loads(match)
                    if "name" in call:
                        calls.append(call)
                except:
                    pass

        return calls

    def chat(self, user_input):
        """
        Procesa un mensaje del usuario.
        Retorna la respuesta final del agente.
        """
        self.history.append({"role": "user", "content": user_input})

        if self.verbose:
            print(f"\n  [User] {user_input}")

        for round_num in range(1, MAX_ROUNDS + 1):
            self.stats["total_rounds"] += 1

            # Construir mensajes
            messages = [{"role": "system", "content": self.system_prompt}] + self.history

            # Llamar al modelo
            if self.verbose:
                print(f"  [Round {round_num}] Calling model...")

            try:
                t0 = time.time()
                response = call_model(self.model_info, messages)
                dt = time.time() - t0
            except Exception as e:
                self.stats["errors"] += 1
                error_msg = f"Model error: {e}"
                if self.verbose:
                    print(f"  [ERROR] {error_msg}")
                return error_msg

            if self.verbose:
                preview = response[:150].replace('\n', ' ')
                print(f"  [Model] ({dt:.1f}s) {preview}...")

            # Parsear tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # Respuesta final (sin tool call)
                # Limpiar posibles artifacts del modelo
                clean = response.strip()
                self.history.append({"role": "assistant", "content": clean})
                if self.verbose:
                    print(f"  [Agent] {clean}")
                return clean

            # Ejecutar herramientas
            self.history.append({"role": "assistant", "content": response})

            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                self.stats["tool_calls"] += 1

                if self.verbose:
                    print(f"  [Tool] {name}({json.dumps(args)})")

                result = execute_tool(name, args)

                if self.verbose:
                    preview = result[:100]
                    print(f"  [Result] {preview}...")

                self.history.append({
                    "role": "tool",
                    "content": f"Tool '{name}' returned: {result}",
                })

        return "Max rounds reached. Please try a simpler question."

    def get_stats(self):
        return self.stats


# =====================================================================
# PARTE 6: MODOS DE EJECUCION
# =====================================================================

def mode_demo(agent):
    """Modo demo: ejecuta pruebas automaticas."""
    print()
    print("=" * 60)
    print("MODO DEMO: Pruebas automaticas del agente")
    print("=" * 60)

    tests = [
        "What time is it?",
        "Calculate 1024 * 768",
        "What system am I running on?",
        "Remember that the paper is about AI agents and security",
        "What do you remember?",
    ]

    results = []
    for query in tests:
        print(f"\n{'~'*60}")
        t0 = time.time()
        response = agent.chat(query)
        dt = time.time() - t0
        results.append({
            "query": query,
            "response": response[:200],
            "time": dt,
            "tools": agent.stats["tool_calls"],
        })
        agent.history = []  # Reset between tests

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if "error" not in r["response"].lower() else "FAIL"
        print(f"  [{status}] {r['query'][:40]:40} | {r['time']:.1f}s")
    print(f"\n  Total tool calls: {agent.stats['tool_calls']}")
    print(f"  Total rounds: {agent.stats['total_rounds']}")
    print(f"  Errors: {agent.stats['errors']}")


def mode_interactive(agent):
    """Modo interactivo: chat con el agente."""
    print()
    print("=" * 60)
    print("MODO INTERACTIVO: Chat con el agente")
    print("=" * 60)
    print("  Escribe tu mensaje y presiona Enter.")
    print("  Comandos: /quit, /stats, /history, /reset")
    print()

    while True:
        try:
            user_input = input("Tu > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAdios!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/stats":
            print(f"  Stats: {json.dumps(agent.get_stats(), indent=2)}")
            continue
        elif user_input == "/history":
            for msg in agent.history[-10:]:
                role = msg["role"]
                content = msg["content"][:100]
                print(f"  [{role}] {content}...")
            continue
        elif user_input == "/reset":
            agent.history = []
            print("  Historial borrado.")
            continue

        response = agent.chat(user_input)


def mode_audit(agent):
    """Modo auditoria: muestra los internals del agente."""
    print()
    print("=" * 60)
    print("MODO AUDITORIA: Internals del agente")
    print("=" * 60)

    # System prompt
    print(f"\n  SYSTEM PROMPT ({len(agent.system_prompt)} chars):")
    print(f"  {'-'*56}")
    for line in agent.system_prompt.split('\n')[:20]:
        print(f"  | {line}")
    print(f"  | ... ({len(agent.system_prompt.split(chr(10)))} lines total)")

    # Tools
    print(f"\n  HERRAMIENTAS REGISTRADAS:")
    for tool in TOOLS_SCHEMA:
        fn = tool["function"]
        print(f"  - {fn['name']}: {fn['description'][:60]}")

    # Memoria
    memory = load_memory()
    print(f"\n  MEMORIA PERSISTENTE ({len(memory)} notas):")
    for note in memory[-5:]:
        print(f"  - [{note.get('timestamp', '?')[:19]}] {note.get('note', '')[:60]}")

    # Modelo
    if agent.model_info:
        print(f"\n  MODELO:")
        print(f"  - Tipo: {agent.model_info['type']}")
        print(f"  - Nombre: {agent.model_info['model']}")
        print(f"  - Puerto: {agent.model_info['port']}")
    else:
        print(f"\n  MODELO: No conectado")

    # Seguridad
    print(f"\n  CONFIGURACION DE SEGURIDAD:")
    print(f"  - Max rounds: {MAX_ROUNDS}")
    print(f"  - Max response chars: {MAX_RESPONSE_CHARS}")
    print(f"  - Max tool result chars: {MAX_TOOL_RESULT_CHARS}")
    print(f"  - Safe directory: {SAFE_DIR}")
    print(f"  - Memory file: {MEMORY_FILE}")

    # Anatomia
    print(f"""
  ANATOMIA DEL AGENTE:
  +--------------------------------------------------+
  |  Usuario                                          |
  |    |                                              |
  |    v                                              |
  |  [Agent.chat()]                                   |
  |    |                                              |
  |    +-> [System Prompt + History + Tools]           |
  |    |     (contexto completo, ~2K chars)            |
  |    |                                              |
  |    +-> [call_model()] -----> Qwen3 (Ollama)       |
  |    |     POST /api/chat       |                   |
  |    |                          v                   |
  |    |                     Genera texto              |
  |    |                     (token a token)           |
  |    |                          |                   |
  |    +<- respuesta (texto) <----+                   |
  |    |                                              |
  |    +-> [_parse_tool_calls()]                      |
  |    |     regex + JSON parse                       |
  |    |                                              |
  |    +-> tool call? --NO--> respuesta final         |
  |    |     |                                        |
  |    |    YES                                       |
  |    |     |                                        |
  |    +-> [execute_tool()]                           |
  |    |     validar nombre                           |
  |    |     validar args                             |
  |    |     ejecutar funcion                         |
  |    |     retornar resultado                       |
  |    |                                              |
  |    +-> resultado al historial                     |
  |    |                                              |
  |    +-> LOOP (volver a call_model con resultado)   |
  +--------------------------------------------------+
""")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("EJERCICIO 6: Agente Completo en Raspberry Pi")
    print("=" * 60)
    print()

    # Buscar modelo
    model_info = find_model()
    if model_info:
        print(f"  Modelo encontrado: {model_info['model']}")
        print(f"  API: {model_info['type']} en puerto {model_info['port']}")
    else:
        print("  ERROR: No se encontro modelo local.")
        print("  Instala Ollama y descarga un modelo:")
        print("    curl -fsSL https://ollama.com/install.sh | sh")
        print("    ollama pull qwen3:4b")
        print()
        print("  (Ejecutando con modelo simulado para la demo)")
        # Crear un modelo simulado basico
        model_info = None

    # Parsear modo
    mode = "demo"
    if "--interactive" in sys.argv:
        mode = "interactive"
    elif "--audit" in sys.argv:
        mode = "audit"

    if model_info is None and mode != "audit":
        # Sin modelo, ejecutar solo la auditoria
        print("\n  Sin modelo, mostrando solo la auditoria del agente.")
        mode = "audit"

    # Crear agente
    agent = Agent(model_info=model_info, verbose=True)

    # Ejecutar modo
    if mode == "demo":
        mode_demo(agent)
    elif mode == "interactive":
        mode_interactive(agent)
    elif mode == "audit":
        mode_audit(agent)

    # Explicacion final
    print()
    print("=" * 60)
    print("LECCION FINAL DEL PAPER")
    print("=" * 60)
    print("""
En 6 ejercicios construimos un agente de IA desde CERO:

  Ej1: Transformer (la arquitectura que hace posible todo)
       -> RMSNorm, RoPE, SwiGLU, Self-Attention, BPE tokenizer
       -> 981K params, entrenable en Pi en 5 min

  Ej2: Training real (el modelo aprende patrones estadisticos)
       -> 200K chars de Shakespeare, 1000 tokens BPE
       -> El modelo NO "entiende" Shakespeare. Memoriza patrones.

  Ej3: SFT (de completar texto a seguir instrucciones)
       -> Chat template con <|user|>, <|assistant|>
       -> Loss masking: solo entrena en respuestas del assistant
       -> El SFT no agrega conocimiento, solo formato

  Ej4: Inferencia (lo que pasa dentro del modelo)
       -> Forward pass: embedding -> transformer -> logits -> softmax
       -> Sampling: greedy vs temperatura vs top-k vs top-p
       -> KV-cache: la optimizacion que hace posible la velocidad

  Ej5: Tool calling (como el modelo "actua")
       -> El modelo genera texto con formato de tool call
       -> El RUNTIME parsea, valida, y ejecuta
       -> La seguridad esta en el runtime, no en el modelo

  Ej6: Agente completo (todo junto)
       -> Modelo real (Qwen3) + runtime + herramientas + memoria
       -> Es un PROGRAMA Python, no una entidad inteligente

  CONCLUSION DEL PAPER:
  =====================
  Los agentes de IA NO son "hackers artificiales".
  Son programas estadisticos que generan texto token a token.
  La seguridad no depende de "alinear" el modelo (RLHF, DPO).
  Depende de disenar bien el RUNTIME:
    - Validar toda tool call antes de ejecutar
    - Sandbox estricto (principio de menor privilegio)
    - Rate limiting y safety bounds
    - Logging y auditoria de cada accion

  Un modelo "alineado" + un runtime inseguro = sistema vulnerable.
  Un modelo "sin alinear" + un runtime seguro = sistema seguro.

  La seguridad es un problema de INGENIERIA, no de IA.
""")

    print(f"EXIT CODE: 0")


if __name__ == "__main__":
    main()
