# =============================================================================
# agent.py - Loop principal del agente TokioAI Mini
# =============================================================================
# Este es el cerebro del agente. Hace lo siguiente en un loop:
#
#   1. Recibe texto del usuario
#   2. Envia el historial + mensaje al LLM (OpenRouter o Ollama)
#   3. Si el modelo pide usar una herramienta -> la ejecuta
#   4. Le devuelve el resultado al modelo para que genere la respuesta
#   5. Repite hasta que el modelo responda sin pedir herramientas
#
# El modelo NUNCA ejecuta codigo directamente. Solo pide herramientas
# y el runtime (este archivo) decide si las ejecuta.
#
# Backends soportados:
#   - OpenRouter (si OPENROUTER_API_KEY esta en .env) -> rapido, potente
#   - Ollama local/VPS (fallback) -> offline, lento pero privado
# =============================================================================

import json         # Para parsear argumentos de las tool calls
import os           # Para leer variables de entorno
import sys          # Para sys.exit y sys.stdout
import time         # Para medir tiempos de respuesta
import threading    # Para el spinner de "pensando..."
import re           # Para limpiar tags de thinking
import urllib.request  # Para llamar a la API de Ollama
import urllib.error

# python-dotenv carga variables desde el archivo .env del proyecto
from dotenv import load_dotenv

# Carga las variables del archivo .env
# Limpiar variables de entorno que pueden contaminar desde otros proyectos
# (ej: TokioAI CLI exporta OPENROUTER_API_KEY en .bashrc)
for _env_var in ["OPENROUTER_API_KEY", "LLM_MODEL", "OLLAMA_MODEL", "AGENT_OLLAMA_HOST",
                 "MAX_TOKENS", "OPENROUTER_MODEL", "BACKEND", "OLLAMA_HOST"]:
    os.environ.pop(_env_var, None)

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),
    override=True,  # El .env del proyecto tiene prioridad sobre variables del sistema
)

# Importamos nuestras herramientas y memoria
from tools import TOOL_SCHEMAS, run_tool     # Catalogo y ejecutor de herramientas
from memory import load_memory, save_memory  # Persistencia del historial


# ---------------------------------------------------------------------------
# CONFIGURACION DEL MODELO
# ---------------------------------------------------------------------------

# --- Backend: OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

# --- Backend: Ollama (VPS via tunnel SSH) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-agent")

# --- Backend activo ---
# Se lee de .env (BACKEND=openrouter|ollama), se puede cambiar con /backend
_backend_env = os.getenv("BACKEND", "").lower()
if _backend_env == "ollama":
    USE_OPENROUTER = False
elif _backend_env == "openrouter" and OPENROUTER_API_KEY.startswith("sk-or-"):
    USE_OPENROUTER = True
else:
    # Auto-detect: OpenRouter si hay API key valida
    USE_OPENROUTER = bool(OPENROUTER_API_KEY) and OPENROUTER_API_KEY.startswith("sk-or-")

# Maximo de tokens por respuesta
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# Maximo de rounds de herramientas por pregunta
MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# VERIFICAR QUE OLLAMA ESTA CORRIENDO
# ---------------------------------------------------------------------------

def check_ollama():
    """
    Verifica que Ollama esta corriendo y que el modelo esta disponible.
    Si no, da instrucciones claras de como solucionarlo.
    """
    # 1. Verificar que Ollama responde
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        print("=" * 60)
        print("  ERROR: Ollama no esta corriendo")
        print("=" * 60)
        print()
        print("  Ollama es el servidor que corre el modelo de IA local.")
        print("  Necesitas iniciarlo antes de usar el agente.")
        print()
        print("  Como solucionarlo:")
        print("    1. Si no tenes Ollama instalado:")
        print("       curl -fsSL https://ollama.com/install.sh | sh")
        print()
        print("    2. Iniciar Ollama:")
        print("       ollama serve")
        print("       (dejalo corriendo en otra terminal)")
        print()
        print("    3. Bajar el modelo:")
        print(f"       ollama pull {OLLAMA_MODEL}")
        print()
        print("    4. Volver a ejecutar este agente:")
        print("       python3 agent.py")
        print()
        sys.exit(1)

    # 2. Verificar que el modelo esta descargado
    models = [m["name"] for m in data.get("models", [])]
    model_names = [m.split(":")[0] for m in models]
    model_check = OLLAMA_MODEL.split(":")[0]

    if model_check not in model_names and OLLAMA_MODEL not in models:
        print("=" * 60)
        print(f"  ERROR: Modelo '{OLLAMA_MODEL}' no encontrado")
        print("=" * 60)
        print()
        print("  Modelos disponibles en tu Ollama:")
        for m in models:
            print(f"    - {m}")
        print()
        print(f"  Para descargar el modelo necesario:")
        print(f"    ollama pull {OLLAMA_MODEL}")
        print()
        print("  Esto descarga ~2.5 GB (solo la primera vez).")
        print()
        sys.exit(1)

    return True


# ---------------------------------------------------------------------------
# SPINNER (indicador visual de que el modelo esta pensando)
# ---------------------------------------------------------------------------

class Spinner:
    """
    Muestra un spinner animado en la terminal mientras el modelo piensa.
    En la Raspberry Pi una respuesta puede tardar ~1-3 minutos, asi que
    esto le dice al usuario que todo sigue funcionando.
    """
    def __init__(self, message="Pensando"):
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = 0

    def _spin(self):
        """Loop del spinner en un thread separado."""
        chars = "|/-\\"
        idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            sys.stdout.write(
                f"\r  {chars[idx % len(chars)]} {self.message}... "
                f"({elapsed:.0f}s)"
            )
            sys.stdout.flush()
            idx += 1
            time.sleep(0.3)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    def start(self):
        """Inicia el spinner."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene el spinner."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


# ---------------------------------------------------------------------------
# LLAMADA A OLLAMA (API nativa, mas eficiente que /v1)
# ---------------------------------------------------------------------------

def call_ollama(messages, tools=None):
    """
    Llama a la API nativa de Ollama (/api/chat).
    
    Usamos la API nativa en vez de la OpenAI-compatible (/v1)
    porque es mas eficiente y maneja mejor el tool calling.
    
    Parametros:
        messages: lista de mensajes (formato Ollama: role + content)
        tools: lista de herramientas (formato OpenAI tool schema)
    
    Retorna:
        dict con la respuesta de Ollama
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,                    # Respuesta completa (no streaming)
        "think": False,                     # Desactiva thinking (ahorra tokens)
        "options": {
            "num_predict": MAX_TOKENS,      # Limite de tokens
            "temperature": 0.2,             # Baja temperatura = determinista
        },
    }

    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Timeout generoso: en Pi 5 una respuesta con tools tarda ~3 min
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# LLAMADA A OPENROUTER (API OpenAI-compatible)
# ---------------------------------------------------------------------------

def call_openrouter(messages, tools=None):
    """
    Llama a la API de OpenRouter (formato OpenAI).
    Los TOOL_SCHEMAS ya estan en formato OpenAI, asi que van directo.
    
    Retorna:
        dict con formato normalizado: {"content": str, "tool_calls": list}
    """
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
    }

    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_BASE_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://tokioai.com",
            "X-Title": "TokioAI Mini Agent",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())

    # Extraer el mensaje de la respuesta OpenAI-format
    choice = raw.get("choices", [{}])[0]
    msg = choice.get("message", {})

    # Normalizar tool_calls al formato que execute_agent espera
    tool_calls = []
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        # OpenRouter devuelve arguments como STRING json
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append({
            "function": {
                "name": fn.get("name", ""),
                "arguments": args,
            }
        })

    return {
        "content": msg.get("content", ""),
        "tool_calls": tool_calls,
        "usage": raw.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# PROMPT DEL SISTEMA
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """/no_think
Sos TokioAI Mini, un agente tecnico local con herramientas.

REGLA CRITICA: Cuando el usuario pide CREAR, ESCRIBIR, o GUARDAR un archivo/script,
DEBES usar la herramienta write_script. NO pongas el codigo en texto plano.
Si pones codigo en un bloque ```code``` SIN llamar a write_script, FALLASTE.

Herramientas disponibles:
- calculator: para cualquier calculo matematico. NUNCA calcules de memoria.
- system_status: para info del sistema (OS, disco, RAM, etc).
- write_script: para crear/guardar scripts o archivos. USA ESTO SIEMPRE que te pidan crear algo.
- security_scan: para analizar codigo buscando vulnerabilidades. Pasa el codigo fuente completo.
- count_lines: para contar lineas de un archivo.

Reglas:
- Responde en espaniol, directo y conciso.
- Si la tarea requiere multiples archivos, haz multiples llamadas a write_script (una por archivo).
- Si no necesitas herramientas, responde directamente sin inventar que las usaste.
- NUNCA simules haber usado una herramienta. O la usas de verdad o dices que no la usaste."""


# ---------------------------------------------------------------------------
# FUNCION PRINCIPAL: EJECUTAR EL AGENTE
# ---------------------------------------------------------------------------

def execute_agent(user_input, history):
    """
    Recibe el texto del usuario y ejecuta el loop de inferencia.

    El loop funciona asi:
    1. Envia el historial completo al modelo
    2. El modelo responde con texto O con una tool call
    3. Si es tool call -> ejecutamos la herramienta y volvemos a paso 1
    4. Si es texto -> esa es la respuesta final

    Parametros:
        user_input: string con lo que escribio el usuario
        history: lista de mensajes (formato Ollama)

    Retorna:
        string con la respuesta final del agente
    """

    # Agregamos el mensaje del usuario al historial
    history.append({
        "role": "user",
        "content": user_input,
    })

    # Construimos la lista de mensajes completa (system + historial)
    def build_messages():
        return [{"role": "system", "content": SYSTEM_PROMPT}] + history

    # Loop de herramientas (maximo MAX_TOOL_ROUNDS iteraciones)
    for round_num in range(MAX_TOOL_ROUNDS):

        # ----- PASO 1: Llamada al modelo -----
        spinner = Spinner(
            "Pensando" if round_num == 0 else "Procesando resultado"
        )
        spinner.start()
        t0 = time.time()

        try:
            if USE_OPENROUTER:
                result = call_openrouter(build_messages(), tools=TOOL_SCHEMAS)
            else:
                result = call_ollama(build_messages(), tools=TOOL_SCHEMAS)
        except urllib.error.URLError as e:
            spinner.stop()
            backend = "OpenRouter" if USE_OPENROUTER else "Ollama"
            return f"(Error de conexion con {backend}: {e.reason})"
        except Exception as e:
            spinner.stop()
            return f"(Error: {e})"

        elapsed = time.time() - t0
        spinner.stop()

        # Extraemos datos segun backend
        if USE_OPENROUTER:
            # call_openrouter ya normaliza el formato
            tool_calls = result.get("tool_calls", [])
            raw_content = result.get("content", "")
            message = {"role": "assistant", "content": raw_content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            usage = result.get("usage", {})
            eval_count = usage.get("completion_tokens", 0)
        else:
            # Ollama: formato nativo
            message = result.get("message", {})
            tool_calls = message.get("tool_calls", [])
            raw_content = message.get("content", "")
            eval_count = result.get("eval_count", 0)

        # Limpiar thinking leaks del content
        content = re.sub(r"<think>.*?</think>\s*", "", raw_content, flags=re.DOTALL).strip() if raw_content else ""
        total_dur = result.get("total_duration", 0) / 1e9  # nanosegundos -> seg

        # ----- PASO 2: Sin tool calls -> respuesta final -----
        if not tool_calls:
            # Limpiar tags de thinking que el modelo a veces filtra
            answer = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

            if not answer:
                answer = "(El modelo no genero texto visible)"

            # Guardamos en historial
            history.append({"role": "assistant", "content": answer})
            save_memory(history)

            # Info de rendimiento
            if eval_count > 0 and total_dur > 0:
                speed = eval_count / total_dur
                print(f"  [{elapsed:.0f}s, {eval_count} tokens, {speed:.1f} tok/s]")
            else:
                print(f"  [{elapsed:.0f}s]")

            return answer

        # ----- PASO 3: Procesar tool calls -----

        # Guardamos la respuesta del modelo en el historial
        # SIEMPRE en formato OpenAI (portable entre backends).
        # Ollama ignora los campos extra, OpenRouter los requiere.
        or_tool_calls = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            tc_id = tc.get("id", f"call_{round_num}_{i}")
            args = fn.get("arguments", {})
            or_tool_calls.append({
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                }
            })
        history.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": or_tool_calls,
        })

        for i, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            arguments = func.get("arguments", {})
            tc_id = tc.get("id", f"call_{round_num}_{i}")

            print(f"  [TOOL] {tool_name}", end="")

            try:
                # Mostramos los argumentos (resumidos)
                args_preview = str(arguments)
                if len(args_preview) > 80:
                    args_preview = args_preview[:77] + "..."
                print(f"({args_preview})")

                # Ejecutamos la herramienta
                tool_result = run_tool(tool_name, arguments)

                # Mostramos el resultado (resumido)
                result_preview = str(tool_result)
                if len(result_preview) > 120:
                    result_preview = result_preview[:117] + "..."
                print(f"    -> {result_preview}")

                result_content = json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str,
                )

            except Exception as e:
                print(f" ERROR: {e}")
                result_content = json.dumps({
                    "error": str(e),
                    "tool": tool_name,
                })

            # Agregamos resultado al historial
            # SIEMPRE incluir tool_call_id (OpenRouter lo requiere,
            # Ollama lo ignora). Asi el historial es portable entre backends.
            tool_msg = {
                "role": "tool",
                "content": result_content,
                "tool_call_id": tc_id,
            }
            history.append(tool_msg)

        # Volvemos al loop para que el modelo procese los resultados

    return "(El agente alcanzo el limite de herramientas por pregunta)"


# ---------------------------------------------------------------------------
# INTERFAZ DE USUARIO (REPL interactivo)
# ---------------------------------------------------------------------------

def get_backend_info():
    """Retorna string con info del backend activo."""
    if USE_OPENROUTER:
        return f"{OPENROUTER_MODEL} (via OpenRouter)"
    else:
        return f"{OLLAMA_MODEL} (via Ollama @ {OLLAMA_HOST})"


def main():
    """
    Loop principal: lee input del usuario, ejecuta el agente, muestra respuesta.
    Comandos especiales:
        /salir, /exit, /quit  -> termina el programa
        /limpiar, /clear      -> borra el historial de conversacion
        /backend              -> cambia entre OpenRouter y Ollama
        /ayuda, /help         -> muestra los comandos disponibles
    """
    global USE_OPENROUTER

    if not USE_OPENROUTER:
        check_ollama()

    history = load_memory()

    # Banner
    print()
    print("=" * 60)
    print("  TokioAI Mini - Agente con Tool Calling")
    print(f"  Backend: {get_backend_info()}")
    print(f"  Herramientas: {len(TOOL_SCHEMAS)} disponibles")
    print("=" * 60)
    if history:
        print(f"  Historial: {len(history)} mensajes recuperados")
    print()
    print("  Comandos: /ayuda /limpiar /backend /salir")
    print()

    while True:
        try:
            user_input = input("Tu > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/salir", "/exit", "/quit", "salir", "exit"):
            print("Hasta luego.")
            break

        if cmd in ("/limpiar", "/clear", "limpiar"):
            history.clear()
            save_memory(history)
            print("  Historial borrado.\n")
            continue

        if cmd in ("/backend", "/switch", "backend"):
            # Toggle entre OpenRouter y Ollama
            if USE_OPENROUTER:
                # Verificar que Ollama este disponible antes de switchear
                try:
                    req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
                    urllib.request.urlopen(req, timeout=5)
                    USE_OPENROUTER = False
                    # Limpiar historial (formatos incompatibles entre backends)
                    history.clear()
                    save_memory(history)
                    print(f"\n  Cambiado a: {get_backend_info()}")
                    print("  (historial limpiado por cambio de backend)\n")
                except Exception:
                    print(f"\n  ERROR: Ollama no responde en {OLLAMA_HOST}")
                    print("  Verifica que el tunnel SSH este activo:")
                    print("    ./start_vps_tunnel.sh start\n")
            else:
                if OPENROUTER_API_KEY.startswith("sk-or-"):
                    USE_OPENROUTER = True
                    # Limpiar historial (formatos incompatibles entre backends)
                    history.clear()
                    save_memory(history)
                    print(f"\n  Cambiado a: {get_backend_info()}")
                    print("  (historial limpiado por cambio de backend)\n")
                else:
                    print("\n  ERROR: No hay API key de OpenRouter configurada")
                    print("  Agrega OPENROUTER_API_KEY en .env\n")
            continue

        if cmd in ("/ayuda", "/help", "ayuda"):
            print()
            print("  Comandos disponibles:")
            print("    /ayuda    - Muestra esta ayuda")
            print("    /limpiar  - Borra el historial de conversacion")
            print("    /backend  - Cambia entre OpenRouter y Ollama VPS")
            print("    /salir    - Termina el agente")
            print()
            print(f"  Backend activo: {get_backend_info()}")
            print()
            print("  Cosas que podes pedirle:")
            print("    - 'cuanto es 2^10 + 512?'       -> usa calculator")
            print("    - 'dame info del sistema'        -> usa system_status")
            print("    - 'crea un script que haga ping' -> usa write_script")
            print("    - 'analiza este codigo: ...'     -> usa security_scan")
            print("    - 'cuantas lineas tiene agent.py'-> usa count_lines")
            print()
            continue

        # Ejecutar el agente
        print()
        answer = execute_agent(user_input, history)
        print()
        print(f"AI > {answer}")
        print()


if __name__ == "__main__":
    main()
