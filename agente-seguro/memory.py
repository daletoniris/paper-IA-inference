# =============================================================================
# memory.py - Memoria persistente del agente TokioAI Mini
# =============================================================================
# Guarda el historial de conversacion en un archivo JSON local.
# Esto permite que el agente recuerde conversaciones anteriores
# sin necesidad de reentrenar el modelo.
# =============================================================================

import json          # Para serializar/deserializar el historial
from pathlib import Path  # Manejo de rutas multiplataforma

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

# Archivo donde se persiste el historial de conversacion
MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"

# Maximo de mensajes que guardamos (evita que el archivo crezca infinito)
# Tambien evita superar el limite de contexto del modelo
MAX_MESSAGES = 20


# ---------------------------------------------------------------------------
# CARGAR HISTORIAL
# ---------------------------------------------------------------------------

def load_memory():
    """
    Lee el historial guardado en disco.
    Si no existe o esta corrupto, devuelve lista vacia (sin crashear).
    Solo devuelve los ultimos MAX_MESSAGES mensajes.
    Sanitiza mensajes para compatibilidad entre backends (OpenRouter/Ollama).
    """

    # Si el archivo no existe, empezamos de cero
    if not MEMORY_FILE.exists():
        return []

    try:
        # Lee el JSON del disco
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))

        # Verifica que sea una lista (formato esperado)
        if isinstance(data, list):
            # Sanitizar: mensajes tool/tool_calls necesitan IDs para OpenRouter
            sanitized = _sanitize_history(data)
            # Devuelve solo los ultimos N mensajes
            return sanitized[-MAX_MESSAGES:]

    except (json.JSONDecodeError, OSError):
        # Si el archivo esta corrupto o no se puede leer, seguimos sin memoria
        pass

    return []


def _sanitize_history(messages):
    """
    Limpia el historial para que sea compatible con ambos backends.
    - Agrega tool_call_id a mensajes tool que no lo tengan
    - Normaliza tool_calls en assistant messages al formato OpenAI
    - Descarta secuencias tool incompletas (tool sin assistant previo)
    """
    result = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "")

        if role == "tool" and "tool_call_id" not in msg:
            # Buscar el assistant message anterior para extraer el ID
            tc_id = _find_tool_call_id(result, msg)
            msg = dict(msg)  # copia para no mutar el original
            msg["tool_call_id"] = tc_id
            result.append(msg)

        elif role == "assistant" and "tool_calls" in msg:
            # Normalizar tool_calls al formato OpenAI
            normalized_tcs = []
            for j, tc in enumerate(msg.get("tool_calls", [])):
                fn = tc.get("function", {})
                tc_id = tc.get("id", f"call_hist_{i}_{j}")
                args = fn.get("arguments", {})
                normalized_tcs.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    }
                })
            msg = dict(msg)
            msg["tool_calls"] = normalized_tcs
            result.append(msg)
        else:
            result.append(msg)

    return result


def _find_tool_call_id(prev_messages, tool_msg):
    """
    Busca hacia atras un assistant message con tool_calls y devuelve
    el ID correspondiente. Si no encuentra, genera uno sintetico.
    """
    # Contar cuantos tool messages hay entre el ultimo assistant y este
    tool_index = 0
    for msg in reversed(prev_messages):
        if msg.get("role") == "tool":
            tool_index += 1
        elif msg.get("role") == "assistant" and "tool_calls" in msg:
            tcs = msg.get("tool_calls", [])
            idx = len(tcs) - 1 - tool_index
            if 0 <= idx < len(tcs):
                return tcs[idx].get("id", f"call_recovered_{idx}")
            break
    return f"call_recovered_{id(tool_msg) % 10000}"


# ---------------------------------------------------------------------------
# GUARDAR HISTORIAL
# ---------------------------------------------------------------------------

def save_memory(messages):
    """
    Guarda el historial en disco, recortando a los ultimos MAX_MESSAGES.
    Usa indent=2 para que sea legible si lo abris manualmente.
    ensure_ascii=False para soportar caracteres en espaniol.
    """

    # Recorta a los ultimos N mensajes antes de guardar
    recent = messages[-MAX_MESSAGES:]

    # Escribe el JSON al disco
    MEMORY_FILE.write_text(
        json.dumps(recent, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
