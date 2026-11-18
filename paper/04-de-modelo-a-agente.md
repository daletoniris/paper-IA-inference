# Parte 4: De Modelo a Agente

## 4.1 Un modelo solo genera texto. Un agente actua.

Esta distincion es la mas importante de todo el paper.

| | Modelo | Agente |
|---|---|---|
| Que hace | Genera texto | Ejecuta acciones |
| Acceso al mundo | Ninguno | A traves de herramientas |
| Quien ejecuta | Nadie (solo produce texto) | El runtime |
| Memoria | Solo el contexto del prompt | Contexto + persistencia |
| Autonomia | Cero (responde y listo) | Puede decidir multiples pasos |

Un modelo es un componente. Un agente es una arquitectura.

```
MODELO SOLO:
  Usuario -> "Que hora es?" -> Modelo -> "No tengo acceso a la hora actual"

AGENTE:
  Usuario -> "Que hora es?" -> Modelo -> tool_call: get_time()
                                           -> Runtime ejecuta -> 15:42:33
                                             -> Modelo recibe resultado
                                               -> "Son las 15:42:33"
```


## 4.2 El runtime: quien realmente ejecuta las cosas

**El runtime es el programa que rodea al modelo.**

El modelo solo genera texto. Si ese texto contiene una tool call,
es el RUNTIME quien:
1. Parsea la tool call (nombre + argumentos)
2. Valida que la herramienta exista y este permitida
3. Valida los argumentos
4. Ejecuta la herramienta
5. Devuelve el resultado al modelo

**El modelo no ejecuta nada. NUNCA.**

```python
# El modelo genera esto (texto en formato JSON):
{
    "tool_call": {
        "name": "execute_command",
        "arguments": {"command": "nmap -sV 10.0.0.1"}
    }
}

# Pero es el RUNTIME quien decide:
# - Existe la herramienta "execute_command"? -> Si/No
# - El usuario tiene permiso para usarla? -> Si/No
# - El argumento es valido? -> Si/No
# - Se ejecuta o se bloquea? -> Decision del runtime, NO del modelo
```

**La seguridad esta en el runtime, no en el modelo.**
Si el runtime permite ejecutar `rm -rf /`, el modelo va a poder
pedir que se ejecute. No importa cuantas veces le digas en el
system prompt "no borres nada".


## 4.3 Tool calling: el protocolo de accion

Tool calling es un protocolo estandarizado donde el modelo expresa
que herramienta quiere usar y con que argumentos.

**Paso 1: Definir las herramientas disponibles**

```python
# Se le dice al modelo que herramientas existen
herramientas = [
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "Obtiene info del sistema operativo",
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
            "description": "Calcula una expresion aritmetica",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Ej: 25*4+10",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]
```

**Paso 2: El modelo "pide" usar una herramienta**

Cuando el modelo recibe un prompt como "Cuanto es 125 * 48?",
y tiene herramientas disponibles, genera una respuesta especial:

```json
{
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": "{\"expression\": \"125 * 48\"}"
            }
        }
    ]
}
```

El modelo NO calcula 125 * 48. Genera el texto `"125 * 48"` como
argumento de la herramienta. El runtime lo ejecuta.

**Paso 3: El runtime ejecuta y devuelve el resultado**

```json
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "{\"expression\": \"125 * 48\", \"result\": 6000}"
}
```

**Paso 4: El modelo recibe el resultado y genera la respuesta final**

```
"El resultado de 125 * 48 es 6000."
```


## 4.4 El loop de inferencia: contexto -> inferencia -> accion -> observacion

El corazon de un agente es un loop:

```
1. CONTEXTO: System prompt + historial + herramientas disponibles
       |
       v
2. INFERENCIA: El modelo procesa todo y genera una respuesta
       |
       v
3. DECISION: La respuesta contiene una tool call?
       |                    |
      NO                   SI
       |                    |
       v                    v
  Respuesta          4. EJECUCION: El runtime ejecuta la herramienta
   final                    |
                            v
                     5. OBSERVACION: El resultado se agrega al contexto
                            |
                            v
                     Volver a paso 2 (nueva inferencia con el resultado)
```

Este loop se repite hasta que:
- El modelo responde sin pedir herramientas (respuesta final)
- Se alcanza un limite de iteraciones (safety)
- El runtime bloquea una accion (seguridad)

```python
def agent_loop(user_input, history, tools, max_rounds=5):
    """
    Loop principal de un agente.
    Exactamente como funciona cualquier agente de IA.
    """

    history.append({"role": "user", "content": user_input})

    for round_num in range(max_rounds):

        # INFERENCIA: enviar todo al modelo
        response = llm.chat(
            messages=[SYSTEM_PROMPT] + history,
            tools=tools,
        )

        message = response.choices[0].message

        # DECISION: hay tool calls?
        if not message.tool_calls:
            # No -> respuesta final
            history.append({"role": "assistant", "content": message.content})
            return message.content

        # Si -> EJECUCION
        history.append(message)  # guardar la tool call en el historial

        for tool_call in message.tool_calls:
            # El RUNTIME valida y ejecuta
            result = runtime.execute(tool_call.function.name,
                                     tool_call.function.arguments)

            # OBSERVACION: resultado al contexto
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Volver al inicio del loop (nueva inferencia con el resultado)

    return "Limite de iteraciones alcanzado."
```


## 4.5 Memoria: contexto vs persistencia

**Contexto** es la informacion que el modelo ve en cada inferencia.
Tiene un limite (la ventana de contexto: 4K, 32K, 128K, 1M tokens).

**Memoria** es persistencia entre sesiones. El modelo no la tiene
nativamente -- es una funcionalidad del runtime.

```python
# La "memoria" de un agente es simplemente un archivo en disco
# que el runtime carga y agrega al prompt.

import json

def cargar_memoria():
    """Lee la memoria guardada en disco."""
    try:
        with open("memoria.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def guardar_memoria(mensajes):
    """Persiste los ultimos N mensajes."""
    with open("memoria.json", "w") as f:
        json.dump(mensajes[-20:], f)

# Cuando el agente arranca, carga la memoria y la agrega al contexto:
history = cargar_memoria()
# Ahora el modelo "recuerda" conversaciones anteriores
# Pero NO aprendio nada nuevo -- solo tiene mas texto en el prompt
```

**Concepto clave:** la memoria del agente NO esta en los pesos del modelo.
Esta en un archivo JSON (o base de datos) que el RUNTIME gestiona.
El modelo solo ve lo que el runtime pone en el prompt.


## 4.6 Codigo: agente minimalista completo

Volvamos al agente minimalista del directorio `paper-IA-inference/`.
Tiene 3 archivos y ~370 lineas. Usa Qwen 3.5 35B via vLLM.

**Arquitectura:**

```
Usuario (consola)
    |
    v
agent.py (runtime -- loop de inferencia)
    |
    +-- tools.py (herramientas: system_status, calculator, write_script)
    |
    +-- memory.py (persistencia: guarda historial en JSON)
    |
    v
Qwen 3.5 35B en vLLM (modelo -- solo genera texto)
```

**Lo que el agente PUEDE hacer (decidido por el runtime):**
1. Consultar info del sistema (system_status)
2. Calcular expresiones aritmeticas (calculator -- con AST seguro)
3. Escribir scripts a disco (write_script -- SIN ejecutarlos)

**Lo que el agente NO PUEDE hacer (bloqueado por el runtime):**
1. Ejecutar comandos del sistema
2. Leer archivos arbitrarios
3. Conectarse a la red
4. Ejecutar scripts que genero
5. Cualquier cosa que no este en TOOL_REGISTRY

La seguridad esta en el TOOL_REGISTRY, no en el system prompt.


## 4.7 Anatomia de una tool call real

Cuando el usuario escribe "Analiza el estado de este equipo",
esto es lo que pasa internamente:

```
1. El runtime construye el prompt:
   [
     {"role": "system", "content": "Sos TokioAI..."},
     {"role": "user", "content": "Analiza el estado de este equipo"}
   ]
   + tools: [system_status, calculator, write_script]

2. Envia todo a Qwen via API OpenAI

3. Qwen genera (token por token):
   tool_calls: [
     {
       "name": "system_status",
       "arguments": "{}"
     }
   ]

4. El runtime intercepta la tool call:
   - "system_status" esta en TOOL_REGISTRY? -> SI
   - Argumentos validos? -> SI (ninguno requerido)
   -> EJECUTA system_status()

5. Resultado real:
   {
     "sistema_operativo": "Linux",
     "version_so": "6.1.0-rpi7-rpi-v8",
     "arquitectura": "aarch64",
     "version_python": "3.11.2",
     "disco_total_gb": 234.57,
     "disco_libre_gb": 189.23,
     "hora_utc": "2026-11-10T21:15:33+00:00"
   }

6. El runtime agrega el resultado al historial y hace otra inferencia

7. Qwen recibe el resultado y genera la respuesta final (token por token):
   "El equipo corre Linux en ARM64 con Python 3.11.
    Tiene 234 GB de disco total con 189 GB libres..."

8. El runtime muestra la respuesta al usuario
```

**El modelo nunca toco el sistema operativo.**
Solo genero el texto "system_status" como herramienta deseada.
El runtime ejecuto la funcion Python y le devolvio el resultado.

---

*En la siguiente seccion: el mismo modelo Qwen, dos agentes
radicalmente distintos. La diferencia es el runtime.*
