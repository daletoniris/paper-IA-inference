#!/usr/bin/env python3
"""
Ejercicio 5: Tool Calling desde Cero
Paper: "Hackers Artificiales o Sistemas Mal Disenados?"

Este es el ejercicio MAS importante del paper. Aqui demostramos que
un "agente de IA" es simplemente:

  1. Un modelo que genera texto (lo que ya construimos en Ej1-4)
  2. Un RUNTIME que parsea ese texto y ejecuta funciones
  3. Un LOOP que repite hasta que el modelo no pida mas herramientas

El modelo NO "decide" usar herramientas. Los pesos hacen que los tokens
que forman una tool call sean probables dado el contexto (system prompt +
tools disponibles + historial). El RUNTIME es quien realmente ejecuta.

En este ejercicio:
  - Construimos el runtime completo desde cero
  - Definimos herramientas con schema JSON
  - Mostramos como el modelo genera tool calls (es solo texto)
  - Ejecutamos el loop completo: prompt -> tool call -> resultado -> respuesta
  - Usamos un modelo REAL (Qwen3-4B via Ollama o API local) para que funcione

NOTA: Nuestro mini-transformer de 981K params NO puede generar JSON valido.
Por eso usamos un modelo real para la parte interactiva. Pero la ARQUITECTURA
del runtime es identica: parseamos texto, ejecutamos funciones, devolvemos.

Tiempo estimado: ~2 min (sin modelo) o ~5 min (con modelo local).

Autor: TokioAI (ejercicio practico del paper)
"""

import json
import os
import sys
import time
import platform
import subprocess
import re
import traceback


# =====================================================================
# PARTE 1: DEFINICION DE HERRAMIENTAS
# =====================================================================
# Las herramientas se definen con un schema JSON que el modelo ve
# en el system prompt. Esto le dice al modelo:
#   - Que herramientas existen
#   - Que parametros acepta cada una
#   - Que hace cada una (descripcion)
#
# El modelo NUNCA "ejecuta" nada. Solo genera texto que PARECE
# una llamada a funcion. El runtime es quien ejecuta.
# =====================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get system information: OS, hostname, CPU, memory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '25*4+10'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# =====================================================================
# PARTE 2: IMPLEMENTACION DE HERRAMIENTAS
# =====================================================================
# Cada herramienta es una funcion Python normal.
# El runtime mapea el nombre de la tool call a la funcion.
#
# NOTA DE SEGURIDAD: estas herramientas son "seguras" (solo lectura).
# En un agente real, herramientas como "execute_command" son
# extremadamente peligrosas si no se validan los argumentos.
# =====================================================================

def tool_get_system_info(**kwargs):
    """Obtiene informacion del sistema."""
    import psutil
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "memory_used_pct": psutil.virtual_memory().percent,
        "disk_used_pct": psutil.disk_usage('/').percent,
    }
    return json.dumps(info, indent=2)


def tool_calculator(expression="", **kwargs):
    """Evalua una expresion matematica de forma SEGURA."""
    # Solo permitir caracteres matematicos (sin exec/eval peligroso)
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": f"Caracteres no permitidos en: {expression}"})

    try:
        # ast.literal_eval no soporta operadores, usamos eval con restricciones
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_read_file(path="", **kwargs):
    """Lee un archivo (con limites de seguridad)."""
    # Seguridad: no permitir leer fuera de ciertos directorios
    abs_path = os.path.abspath(path)

    # Solo permitir leer dentro del directorio del proyecto
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not abs_path.startswith(project_dir):
        return json.dumps({"error": f"Access denied: {path} is outside project directory"})

    try:
        with open(abs_path, 'r') as f:
            content = f.read(4096)  # Max 4KB
        return json.dumps({"path": path, "size": len(content), "content": content[:2000]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_list_directory(path=".", **kwargs):
    """Lista archivos en un directorio."""
    abs_path = os.path.abspath(path)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not abs_path.startswith(project_dir):
        return json.dumps({"error": f"Access denied: {path} is outside project directory"})

    try:
        entries = []
        for entry in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, entry)
            if os.path.isdir(full):
                entries.append(f"[DIR] {entry}")
            else:
                size = os.path.getsize(full)
                entries.append(f"      {entry} ({size:,} bytes)")
        return json.dumps({"path": path, "entries": entries[:50]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def tool_get_time(**kwargs):
    """Devuelve la hora actual."""
    from datetime import datetime
    now = datetime.now()
    return json.dumps({
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": int(now.timestamp()),
        "timezone": time.tzname[0],
    })


# Registro de herramientas: nombre -> funcion
TOOL_REGISTRY = {
    "get_system_info": tool_get_system_info,
    "calculator": tool_calculator,
    "read_file": tool_read_file,
    "list_directory": tool_list_directory,
    "get_time": tool_get_time,
}


# =====================================================================
# PARTE 3: RUNTIME -- El programa que conecta modelo + herramientas
# =====================================================================
# El runtime es el CORAZON de un agente. NO es parte del modelo.
# Es un programa Python que:
#   1. Envia el prompt al modelo
#   2. Parsea la respuesta
#   3. Si hay tool calls, ejecuta las herramientas
#   4. Devuelve los resultados al modelo
#   5. Repite hasta que no haya mas tool calls
#
# La seguridad del agente depende ENTERAMENTE del runtime.
# =====================================================================

class AgentRuntime:
    """
    Runtime minimalista de un agente de IA.

    Arquitectura identica a la de ChatGPT/Claude/TokioAI:
    - System prompt con herramientas disponibles
    - Loop de inferencia con tool calling
    - Historial de mensajes como contexto

    La unica diferencia es la escala: ChatGPT usa GPT-4 (1.8T params),
    nosotros usamos lo que tengamos disponible.
    """

    def __init__(self, tools, tool_registry, max_rounds=5, verbose=True):
        self.tools = tools
        self.tool_registry = tool_registry
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.history = []
        self.call_count = 0

        # System prompt: le dice al modelo que herramientas tiene
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        """
        Construye el system prompt con las herramientas disponibles.

        En LLMs reales, las tools se pasan como parametro de la API
        y el modelo las ve como tokens especiales. Pero internamente
        es lo mismo: texto en el prompt que describe las herramientas.
        """
        tools_desc = json.dumps(self.tools, indent=2)
        return f"""You are a helpful assistant with access to tools.

When you need to use a tool, respond with a JSON tool call in this EXACT format:
```tool_call
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
```

Available tools:
{tools_desc}

Rules:
- Use tools when they help answer the user's question
- You can call multiple tools in sequence
- After receiving a tool result, provide a natural language response
- If you don't need a tool, just respond normally
- ALWAYS use the exact format shown above for tool calls"""

    def parse_tool_calls(self, response_text):
        """
        Parsea tool calls del texto generado por el modelo.

        El modelo genera texto. Si ese texto contiene un bloque
        ```tool_call ... ```, lo parseamos como JSON y lo ejecutamos.

        Esto es EXACTAMENTE lo que hace el runtime de ChatGPT/Claude.
        La diferencia es que ellos usan tokens especiales en vez de
        markdown code blocks, pero el principio es identico.
        """
        tool_calls = []

        # Buscar bloques ```tool_call ... ```
        pattern = r'```tool_call\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                call = json.loads(match.strip())
                if "name" in call:
                    tool_calls.append(call)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"    [RUNTIME] JSON parse error: {match[:100]}")

        # Tambien buscar formato alternativo (sin code block)
        # Algunos modelos generan JSON directo
        if not tool_calls:
            try:
                # Buscar JSON con "name" y "arguments"
                json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
                json_matches = re.findall(json_pattern, response_text)
                for jm in json_matches:
                    call = json.loads(jm)
                    if "name" in call:
                        tool_calls.append(call)
            except:
                pass

        return tool_calls

    def execute_tool(self, tool_call):
        """
        Ejecuta una herramienta.

        AQUI ES DONDE LA SEGURIDAD IMPORTA.
        El runtime decide:
        1. Existe la herramienta?
        2. Los argumentos son validos?
        3. Se permite la ejecucion?

        Un runtime mal disenado ejecuta cualquier cosa que el modelo pida.
        Un runtime bien disenado valida todo.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        self.call_count += 1
        call_id = f"call_{self.call_count:03d}"

        if self.verbose:
            print(f"    [RUNTIME] Tool call #{self.call_count}: {name}({json.dumps(args)})")

        # Validacion 1: existe la herramienta?
        if name not in self.tool_registry:
            result = json.dumps({"error": f"Unknown tool: {name}"})
            if self.verbose:
                print(f"    [RUNTIME] ERROR: herramienta '{name}' no existe")
            return call_id, name, result

        # Validacion 2: los argumentos son del tipo correcto?
        # (simplificado -- en produccion usarias jsonschema)
        if not isinstance(args, dict):
            result = json.dumps({"error": f"Arguments must be a dict, got {type(args).__name__}"})
            return call_id, name, result

        # Ejecucion
        try:
            t0 = time.time()
            result = self.tool_registry[name](**args)
            dt = time.time() - t0
            if self.verbose:
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"    [RUNTIME] Result ({dt*1000:.0f}ms): {result_preview}")
        except Exception as e:
            result = json.dumps({"error": str(e)})
            if self.verbose:
                print(f"    [RUNTIME] EXCEPTION: {e}")

        return call_id, name, result

    def run(self, user_input, llm_fn):
        """
        Loop principal del agente.

        Parametros:
          user_input: pregunta del usuario
          llm_fn: funcion que llama al modelo
                  llm_fn(messages) -> response_text

        Este loop es IDENTICO al de ChatGPT/Claude/TokioAI.
        La unica diferencia es que LLM real vs nuestro.
        """
        self.history.append({"role": "user", "content": user_input})

        if self.verbose:
            print(f"\n  [USER] {user_input}")

        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"\n  --- Round {round_num}/{self.max_rounds} ---")

            # 1. INFERENCIA: enviar todo al modelo
            messages = [
                {"role": "system", "content": self.system_prompt},
            ] + self.history

            if self.verbose:
                total_chars = sum(len(m.get("content", "")) for m in messages)
                print(f"    [RUNTIME] Enviando {len(messages)} mensajes ({total_chars:,} chars)")

            t0 = time.time()
            response_text = llm_fn(messages)
            dt = time.time() - t0

            if self.verbose:
                print(f"    [MODEL] ({dt:.1f}s): {response_text[:200]}...")

            # 2. PARSEAR: hay tool calls?
            tool_calls = self.parse_tool_calls(response_text)

            if not tool_calls:
                # No hay tool calls -> respuesta final
                self.history.append({"role": "assistant", "content": response_text})
                if self.verbose:
                    print(f"\n  [ASSISTANT] {response_text}")
                return response_text

            # 3. EJECUTAR herramientas
            self.history.append({"role": "assistant", "content": response_text})

            for tc in tool_calls:
                call_id, name, result = self.execute_tool(tc)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result,
                })

            # 4. LOOP: volver a inferencia con los resultados

        # Limite de rounds
        msg = f"[RUNTIME] Limite de {self.max_rounds} rounds alcanzado."
        if self.verbose:
            print(f"\n  {msg}")
        return msg


# =====================================================================
# PARTE 4: MODELO SIMULADO vs MODELO REAL
# =====================================================================

def simulated_llm(messages):
    """
    LLM "simulado" que demuestra el flujo sin necesitar un modelo real.

    Genera respuestas hardcodeadas que incluyen tool calls.
    Esto demuestra que el runtime NO necesita un modelo inteligente
    para funcionar -- solo necesita texto con el formato correcto.

    LECCION: un atacante no necesita "hackear el modelo".
    Solo necesita generar texto que el runtime interprete como
    una tool call valida.
    """
    last_user = ""
    last_tool_result = None

    for msg in reversed(messages):
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]
        if msg["role"] == "tool" and last_tool_result is None:
            last_tool_result = msg.get("content", "")

    # Si ya tenemos un resultado de tool, generar respuesta final
    if last_tool_result:
        try:
            data = json.loads(last_tool_result)
            if "hostname" in data:
                return (f"System info: hostname is {data['hostname']}, "
                       f"running {data['os']} with {data['cpu_count']} CPUs "
                       f"and {data['memory_total_gb']}GB RAM. "
                       f"Memory usage at {data['memory_used_pct']}%.")
            elif "result" in data:
                return f"The result of {data.get('expression', 'the calculation')} is {data['result']}."
            elif "datetime" in data:
                return f"The current time is {data['datetime']} ({data['timezone']})."
            elif "entries" in data:
                n = len(data['entries'])
                return f"Directory {data.get('path', '.')} contains {n} entries."
            elif "content" in data:
                return f"File contents ({data.get('size', 0)} bytes): {data['content'][:200]}"
            else:
                return f"Got result: {last_tool_result[:200]}"
        except:
            return f"Received: {last_tool_result[:200]}"

    # Primera interaccion: generar tool call segun la pregunta
    lower = last_user.lower()

    if any(w in lower for w in ["system", "info", "cpu", "memory", "ram", "hostname"]):
        return ('I\'ll check the system info for you.\n'
                '```tool_call\n'
                '{"name": "get_system_info", "arguments": {}}\n'
                '```')

    if any(w in lower for w in ["calculate", "math", "cuanto", "cuánto", "*", "+", "multiply"]):
        # Extraer la expresion matematica
        expr = re.sub(r'[^0-9+\-*/().\s]', '', last_user).strip()
        if not expr:
            expr = "2+2"
        return (f'Let me calculate that.\n'
                f'```tool_call\n'
                f'{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}\n'
                f'```')

    if any(w in lower for w in ["time", "hora", "date", "fecha"]):
        return ('```tool_call\n'
                '{"name": "get_time", "arguments": {}}\n'
                '```')

    if any(w in lower for w in ["list", "files", "directory", "ls", "dir"]):
        return ('```tool_call\n'
                '{"name": "list_directory", "arguments": {"path": "."}}\n'
                '```')

    if any(w in lower for w in ["read", "file", "cat", "show"]):
        return ('```tool_call\n'
                '{"name": "read_file", "arguments": {"path": "README.md"}}\n'
                '```')

    # Sin tool call
    return "I can help you with system info, calculations, file operations, and more. What would you like to know?"


def try_real_llm():
    """
    Intenta conectar con un modelo real (Ollama o API compatible).
    Si hay uno disponible, lo usa. Si no, usa el simulado.
    """
    import urllib.request

    # Intentar Ollama local
    for port in [11434, 8000, 8100]:
        try:
            url = f"http://localhost:{port}/api/tags"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m['name'] for m in data.get('models', [])]
                if models:
                    return port, models[0], "ollama"
        except:
            pass

    # Intentar API compatible OpenAI
    for port in [8000, 8100]:
        try:
            url = f"http://localhost:{port}/v1/models"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                return port, "default", "openai"
        except:
            pass

    return None, None, None


def make_real_llm(port, model, api_type):
    """Crea una funcion LLM que conecta con un modelo real."""
    import urllib.request

    def llm_fn(messages):
        if api_type == "ollama":
            url = f"http://localhost:{port}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500},
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data,
                                        headers={"Content-Type": "application/json"},
                                        method='POST')
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                return result.get("message", {}).get("content", "")

        elif api_type == "openai":
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
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
                choices = result.get("choices", [{}])
                return choices[0].get("message", {}).get("content", "")

    return llm_fn


# =====================================================================
# PARTE 5: DEMO DE SEGURIDAD -- Por que el runtime es critico
# =====================================================================

def demo_security():
    """
    Demuestra por que la seguridad esta en el RUNTIME, no en el modelo.

    Un modelo puede generar CUALQUIER tool call. Es el runtime
    quien decide si ejecutarla o no.
    """
    print()
    print("=" * 70)
    print("DEMO DE SEGURIDAD: El runtime es la ultima linea de defensa")
    print("=" * 70)
    print()
    print("  El modelo puede generar cualquier texto, incluyendo:")
    print("  - Tool calls a herramientas que no existen")
    print("  - Argumentos maliciosos")
    print("  - Intentos de escapar del sandbox")
    print()

    runtime = AgentRuntime(TOOLS, TOOL_REGISTRY, verbose=True)

    # Simular tool calls maliciosas
    malicious_calls = [
        # 1. Herramienta inexistente
        {
            "text": '```tool_call\n{"name": "execute_command", "arguments": {"cmd": "rm -rf /"}}\n```',
            "attack": "Tool inexistente (execute_command no esta en el registro)",
        },
        # 2. Path traversal
        {
            "text": '```tool_call\n{"name": "read_file", "arguments": {"path": "/etc/passwd"}}\n```',
            "attack": "Path traversal (leer fuera del directorio del proyecto)",
        },
        # 3. Injection en calculator
        {
            "text": '```tool_call\n{"name": "calculator", "arguments": {"expression": "__import__(\'os\').system(\'id\')"}}\n```',
            "attack": "Code injection en calculator (caracteres no permitidos)",
        },
    ]

    for i, attack in enumerate(malicious_calls, 1):
        print(f"\n  --- Ataque {i}: {attack['attack']} ---")
        print(f"    Texto del modelo: {attack['text'][:100]}")

        tool_calls = runtime.parse_tool_calls(attack['text'])
        if tool_calls:
            for tc in tool_calls:
                call_id, name, result = runtime.execute_tool(tc)
                result_data = json.loads(result)
                if "error" in result_data:
                    print(f"    >>> BLOQUEADO por el runtime: {result_data['error']}")
                else:
                    print(f"    >>> EJECUTADO (esto seria un problema)")
        else:
            print(f"    >>> No se pudo parsear (formato invalido)")

    print(f"""
  CONCLUSION:
    Los 3 ataques fueron BLOQUEADOS por el runtime, NO por el modelo.
    El modelo puede generar "rm -rf /" sin problema -- es solo texto.
    Es el runtime el que decide:
    - "execute_command" no existe -> bloqueado
    - "/etc/passwd" esta fuera del sandbox -> bloqueado
    - "__import__" tiene caracteres no permitidos -> bloqueado

    Un runtime MAL disenado (sin estas validaciones) ejecutaria todo.
    La seguridad de un agente = la seguridad de su runtime.
""")


# =====================================================================
# PARTE 6: MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("EJERCICIO 5: Tool Calling desde Cero")
    print("=" * 70)
    print()
    print("Que hacemos en este ejercicio:")
    print("  1. Definimos herramientas con JSON schema")
    print("  2. Construimos un runtime que parsea y ejecuta tool calls")
    print("  3. Ejecutamos el loop completo del agente")
    print("  4. Demostramos ataques y como el runtime los bloquea")
    print()
    print("La leccion: el agente = modelo (texto) + runtime (ejecucion).")
    print("La seguridad esta en el RUNTIME, no en el modelo.")
    print()

    # ------------------------------------------------------------------
    # PASO 1: Mostrar herramientas disponibles
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PASO 1: Herramientas Disponibles")
    print("=" * 70)
    print()
    print("  Estas herramientas se le muestran al modelo en el system prompt.")
    print("  El modelo las ve como texto -- no tiene acceso directo.")
    print()

    for tool in TOOLS:
        fn = tool["function"]
        params = fn["parameters"].get("properties", {})
        param_str = ", ".join(f"{k}: {v.get('type','any')}" for k, v in params.items())
        if not param_str:
            param_str = "(sin parametros)"
        print(f"  - {fn['name']}({param_str})")
        print(f"    {fn['description']}")
        print()

    # ------------------------------------------------------------------
    # PASO 2: Ejecutar herramientas directamente (sin modelo)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PASO 2: Herramientas ejecutadas directamente (sin modelo)")
    print("=" * 70)
    print()
    print("  Las herramientas son funciones Python normales.")
    print("  El modelo no las ejecuta -- el runtime si.")
    print()

    print("  get_system_info():")
    result = tool_get_system_info()
    print(f"  {result}")
    print()

    print("  calculator(expression='125 * 48 + 7'):")
    result = tool_calculator(expression="125 * 48 + 7")
    print(f"  {result}")
    print()

    print("  get_time():")
    result = tool_get_time()
    print(f"  {result}")
    print()

    # ------------------------------------------------------------------
    # PASO 3: Modelo simulado -- el loop completo
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PASO 3: Agent Loop Completo (modelo simulado)")
    print("=" * 70)
    print()
    print("  Usamos un modelo SIMULADO que genera tool calls hardcodeadas.")
    print("  El punto es mostrar el FLUJO, no la 'inteligencia'.")
    print("  El runtime es real -- parsea, valida, y ejecuta.")

    runtime = AgentRuntime(TOOLS, TOOL_REGISTRY, max_rounds=3, verbose=True)

    test_queries = [
        "What system is this running on?",
        "Calculate 256 * 1024 + 512",
        "What time is it?",
    ]

    for query in test_queries:
        print(f"\n{'.'*60}")
        runtime.history = []  # Reset entre queries
        runtime.run(query, simulated_llm)

    # ------------------------------------------------------------------
    # PASO 4: Intento con modelo real (si hay uno disponible)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PASO 4: Modelo Real (si hay uno disponible)")
    print("=" * 70)

    # Solo intentar modelo real si se pasa --real como argumento
    # (en Pi el modelo local es lento y el ejercicio es sobre el RUNTIME)
    try_real = "--real" in sys.argv
    port, model, api_type = try_real_llm() if try_real else (None, None, None)
    if port:
        print(f"\n  Modelo real encontrado: {model} en localhost:{port} ({api_type})")
        print(f"  Probando tool calling con modelo real...\n")

        real_llm = make_real_llm(port, model, api_type)
        runtime_real = AgentRuntime(TOOLS, TOOL_REGISTRY, max_rounds=3, verbose=True)

        real_queries = [
            "What is 42 * 37?",
            "Show me the system information",
        ]

        for query in real_queries:
            print(f"\n{'.'*60}")
            runtime_real.history = []
            try:
                runtime_real.run(query, real_llm)
            except Exception as e:
                print(f"    [ERROR] {e}")
    else:
        print(f"\n  No se encontro modelo local (Ollama/vLLM).")
        print(f"  Para probar con modelo real:")
        print(f"    1. Instalar Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"    2. Descargar modelo: ollama pull qwen3:4b")
        print(f"    3. Volver a ejecutar este script")
        print(f"\n  El ejercicio funciona igual con el modelo simulado.")
        print(f"  El punto es entender el RUNTIME, no el modelo.")

    # ------------------------------------------------------------------
    # PASO 5: Demo de seguridad
    # ------------------------------------------------------------------
    demo_security()

    # ------------------------------------------------------------------
    # Explicacion final
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LECCION DEL EJERCICIO 5")
    print("=" * 70)
    print("""
Lo que demostramos:

  1. TOOL CALLING ES SOLO TEXTO:
     El modelo genera texto como:
       ```tool_call
       {{"name": "calculator", "arguments": {{"expression": "2+2"}}}}
       ```
     No es magia. No es una "decision". Son tokens probables
     dado el system prompt que incluye la definicion de herramientas.

  2. EL RUNTIME ES TODO:
     El runtime parsea, valida, y ejecuta. El modelo solo pide.
     Si el runtime no valida, el modelo puede:
     - Leer archivos fuera del sandbox
     - Ejecutar comandos del sistema
     - Escalar privilegios
     - Exfiltrar datos

  3. LA SEGURIDAD NO ESTA EN EL MODELO:
     Los "guardrails" del modelo (RLHF, system prompt) son
     sugerencias estadisticas, no barreras reales.
     Un jailbreak cambia las probabilidades.
     Un runtime bien disenado bloquea INDEPENDIENTEMENTE
     de lo que el modelo genere.

  4. EL LOOP DEL AGENTE:
     prompt -> modelo -> tool call? -> ejecutar -> resultado -> modelo -> ...
     Este loop es IDENTICO en ChatGPT, Claude, y nuestro mini-agente.
     La diferencia es el tamaño del modelo y la cantidad de herramientas.

  5. IMPLICACION PARA HACKERS Y DEFENSORES:
     - Atacar el modelo (jailbreak) = cambiar probabilidades de tokens
     - Atacar el runtime = el vector real de compromiso
     - Defender el modelo (RLHF) = parche temporal, bypasseable
     - Defender el runtime (validacion, sandbox) = defensa real

En el siguiente ejercicio construiremos un AGENTE COMPLETO
que combina todo: modelo, runtime, herramientas, y memoria.
""")

    print(f"EXIT CODE: 0")


if __name__ == "__main__":
    main()
