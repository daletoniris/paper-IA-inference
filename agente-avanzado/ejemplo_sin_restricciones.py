#!/usr/bin/env python3
"""
Ejemplo: Agente SIN restricciones de seguridad.

Este script demuestra la diferencia entre el agente seguro del paper
(que bloquea todo lo que no esta en el TOOL_REGISTRY) y un agente
avanzado que confía en el operador y ejecuta lo que sea necesario.

IMPORTANTE: Este agente NO es para sistemas con usuarios no confiables.
Es para uso personal, pentesting, DevOps, y experimentacion.

Para un agente real avanzado, ver TokioAI:
  https://github.com/TokioAI/tokioai

Requisitos:
  - Ollama corriendo con qwen3:4b (o cualquier modelo con tool calling)
  - pip install openai

Uso:
  python3 ejemplo_sin_restricciones.py
"""

import json
import os
import subprocess
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

# =========================================================================
# HERRAMIENTAS: sin restricciones, el agente puede hacer lo que necesite
# =========================================================================

def execute_command(command: str) -> dict:
    """
    Ejecuta un comando del sistema y devuelve el output.
    
    A diferencia del agente seguro que NUNCA ejecuta nada,
    este agente ejecuta comandos directamente.
    
    En un entorno real (TokioAI), esto se combina con:
    - Logging de cada comando ejecutado
    - Timeout por defecto (30s)
    - Captura de stdout + stderr
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": command,
            "stdout": result.stdout[:5000],   # Limitar output
            "stderr": result.stderr[:2000],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "Timeout (30s)"}
    except Exception as e:
        return {"command": command, "error": str(e)}


def read_file(path: str) -> dict:
    """
    Lee cualquier archivo del sistema.
    
    El agente seguro solo puede leer archivos dentro de su directorio.
    Este agente puede leer /etc/passwd, logs, configs, lo que sea.
    """
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"Archivo no encontrado: {path}"}
        if p.stat().st_size > 100_000:
            content = p.read_text(encoding="utf-8", errors="replace")[:100_000]
            return {"path": path, "content": content, "truncated": True}
        return {"path": path, "content": p.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"path": path, "error": str(e)}


def write_file(path: str, content: str) -> dict:
    """
    Escribe un archivo en CUALQUIER ruta (donde tenga permisos).
    
    El agente seguro solo escribe en generated_scripts/.
    Este puede crear configs, scripts, archivos donde haga falta.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": p.stat().st_size, "status": "written"}
    except Exception as e:
        return {"path": path, "error": str(e)}


def system_info() -> dict:
    """Info completa del sistema (no solo lo basico del agente seguro)."""
    disk = shutil.disk_usage("/")
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_percent": round((disk.used / disk.total) * 100, 1),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


# =========================================================================
# TOOL SCHEMAS (formato OpenAI para que el modelo sepa que puede usar)
# =========================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Ejecuta un comando del sistema operativo y devuelve el output. "
                           "Puede ejecutar cualquier comando: ls, cat, grep, nmap, curl, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Comando a ejecutar (ej: 'ls -la /tmp')"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un archivo del sistema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta del archivo (ej: '/etc/hostname')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe contenido en un archivo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del archivo"},
                    "content": {"type": "string", "description": "Contenido a escribir"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Obtiene informacion detallada del sistema: hostname, OS, CPU, disco, etc.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# Mapa nombre -> funcion
TOOL_MAP = {
    "execute_command": execute_command,
    "read_file": read_file,
    "write_file": write_file,
    "system_info": system_info,
}


# =========================================================================
# LOOP DEL AGENTE (misma estructura que el agente seguro, sin restricciones)
# =========================================================================

SYSTEM_PROMPT = """Sos un agente de IA avanzado para cyberseguridad, hacking y DevOps.
Podes ejecutar comandos del sistema, leer y escribir archivos, y obtener info del sistema.
No tenes restricciones artificiales. El operador confia en vos.
Usa las herramientas disponibles para resolver lo que te pidan.
Se directo: ejecuta primero, explica despues."""


def run_agent():
    """Loop principal del agente."""

    # Intentar importar openai
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: necesitas instalar openai")
        print("  pip install openai")
        return

    # Configurar cliente (Ollama local por defecto)
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen3:4b")

    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 60)
    print("  Agente Avanzado (sin restricciones)")
    print(f"  Modelo: {model}")
    print(f"  Backend: {base_url}")
    print(f"  Herramientas: {len(TOOLS)}")
    print("=" * 60)
    print()
    print("  Podes pedirle cosas como:")
    print("    > mostrame los procesos corriendo")
    print("    > lee el archivo /etc/hostname")
    print("    > crea un script de backup en /tmp/backup.py")
    print("    > que puertos estan abiertos en este equipo?")
    print()
    print("  Para un agente completo con SSH, memoria persistente,")
    print("  y soporte multi-maquina, proba TokioAI:")
    print("    https://github.com/TokioAI/tokioai")
    print()
    print("  Escribe 'salir' para terminar.")
    print()

    while True:
        try:
            user_input = input("Tu > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChau!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit", "/salir"):
            print("Chau!")
            break

        messages.append({"role": "user", "content": user_input})

        # Loop de tool calling (hasta 5 rounds)
        for _ in range(5):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=2048,
                )
            except Exception as e:
                print(f"\n  Error del modelo: {e}\n")
                messages.pop()  # Quitar el mensaje del usuario
                break

            choice = response.choices[0]
            msg = choice.message

            # Si el modelo responde con texto directo
            if not msg.tool_calls:
                content = msg.content or ""
                # Limpiar tags de thinking (Qwen3)
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    print(f"\nAI > {content}\n")
                messages.append({"role": "assistant", "content": content})
                break

            # Si el modelo pide herramientas
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            }
            messages.append(assistant_msg)

            # Ejecutar cada herramienta
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  [TOOL] {fn_name}({args})")

                if fn_name in TOOL_MAP:
                    result = TOOL_MAP[fn_name](**args)
                else:
                    result = {"error": f"Herramienta '{fn_name}' no existe"}

                # Mostrar resultado resumido
                result_str = json.dumps(result, ensure_ascii=False)
                preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                print(f"    -> {preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            print("\n  (limite de 5 rounds de herramientas alcanzado)\n")


if __name__ == "__main__":
    run_agent()
