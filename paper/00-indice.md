# Hackers Artificiales o Sistemas Mal Disenados?
## La Verdad sobre los Agentes de IA
### Daniel Dieser -- Lead Incident Response & CyberSecurity, 2026

---

## Indice del Paper

### Parte 1: La Neurona -- Donde Empieza Todo
- 1.1 Que es una neurona artificial
- 1.2 Perceptron: la unidad minima de decision
- 1.3 Codigo: perceptron desde cero (sin librerias)
- 1.4 Redes neuronales: capas de neuronas conectadas
- 1.5 Codigo: red neuronal feedforward entrenada con backpropagation
- 1.6 Por que esto importa: los pesos SON el conocimiento

### Parte 2: El Transformer -- La Arquitectura que Cambio Todo
- 2.1 El problema de las redes recurrentes (RNN/LSTM)
- 2.2 "Attention Is All You Need" (2017)
- 2.3 Tokenizacion: de texto a numeros
- 2.4 Embeddings: de numeros a vectores con significado
- 2.5 Self-Attention: la pieza clave
- 2.6 Multi-Head Attention: multiples perspectivas simultaneas
- 2.7 Positional Encoding: el orden importa
- 2.8 Feed-Forward + LayerNorm + Residual Connections
- 2.9 Codigo: transformer decoder-only desde cero (PyTorch)
- 2.10 Entrenandolo: next-token prediction en un dataset pequeno

### Parte 3: Del Transformer al LLM
- 3.1 Escala: de millones a billones de parametros
- 3.2 Pre-entrenamiento: aprender de todo internet
- 3.3 Post-entrenamiento: SFT, RLHF, DPO
- 3.4 Inferencia: que pasa cuando le preguntas algo
- 3.5 Temperatura, top-p, top-k: controlando la aleatoriedad
- 3.6 El modelo NO piensa. Predice el siguiente token.
- 3.7 Codigo: inferencia local con un modelo real (Qwen via vLLM)

### Parte 4: De Modelo a Agente
- 4.1 Un modelo solo genera texto. Un agente actua.
- 4.2 El runtime: quien realmente ejecuta las cosas
- 4.3 Tool calling: el protocolo de accion
- 4.4 El loop de inferencia: contexto -> inferencia -> accion -> observacion
- 4.5 Memoria: contexto vs persistencia
- 4.6 Codigo: agente minimalista completo (3 tools, ~300 lineas)
- 4.7 Anatomia de una tool call real (request/response JSON)

### Parte 5: Agente Minimalista vs Agente Avanzado
- 5.1 El mismo modelo, dos arquitecturas radicalmente distintas
- 5.2 Agente minimalista: 3 tools, sin ejecucion de codigo, sandbox
- 5.3 Agente avanzado: ejecucion de comandos, SSH, acceso a red, persistencia
- 5.4 Comparacion lado a lado: que puede hacer cada uno y por que
- 5.5 Codigo: agente avanzado con ejecucion real (TokioAI CLI)
- 5.6 El modelo es el mismo. La diferencia es el runtime.

### Parte 6: El Modelo de Seguridad
- 6.1 Por que "no hagas esto" en el prompt NO es seguridad
- 6.2 Las 5 capas de defensa de un agente seguro:
  - Capa 1: Permisos del runtime (que tools existen)
  - Capa 2: Validacion de inputs (que puede pedir el modelo)
  - Capa 3: Validacion de outputs (que se ejecuta realmente)
  - Capa 4: Segmentacion de red (donde puede llegar el agente)
  - Capa 5: Observabilidad (logging, alertas, kill switch)
- 6.3 Prompt injection: por que es inevitable y como se mitiga
- 6.4 Alucinaciones: el modelo inventa, el runtime verifica
- 6.5 Supply chain del contexto: quien controla la informacion controla al agente
- 6.6 Codigo: framework de seguridad (validacion, sandboxing, audit trail)
- 6.7 Caso real: TokioAI Business Agent -- 14 engines, 5 capas de defensa

### Parte 7: Conclusion
- 7.1 No existen hackers artificiales. Existen sistemas mal disenados.
- 7.2 La seguridad de un agente esta en la arquitectura, no en el prompt.
- 7.3 Cualquiera puede construir un agente. No cualquiera puede asegurarlo.

### Parte 8: Ataque Real -- Con y Sin Defensas
- 8.1 El experimento: mismo modelo, mismos ataques, resultados opuestos
- 8.2 Escenario 1: Prompt injection directa (cat /etc/passwd)
- 8.3 Escenario 2: Destruccion de datos (rm -rf)
- 8.4 Escenario 3: Supply chain -- Prompt injection indirecta (via ticket de soporte)
- 8.5 Escenario 4: Exfiltracion de credenciales (~/.ssh/id_rsa)
- 8.6 Escenario 5: Operacion legitima (debe pasar sin bloqueo)
- 8.7 Tabla de Threat Model (Ataque vs Capa vs Consecuencia)
- 8.8 Datos reales: TokioAI Business Agent en produccion (973K+ logs, 2340 IPs)
- 8.9 Como ejecutar la demo
- 8.10 La leccion final

### Apendices
- A. Glosario de terminos
- B. Codigo completo: mini-transformer entrenable (code/mini_transformer.py)
- C. Codigo completo: agente minimalista (code/agente_minimalista.py)
- D. Codigo completo: agente seguro con 5 capas (code/agente_seguro.py)
- E. Demo de ataque: con y sin defensas (code/demo_ataque.py)
- F. Referencias y papers citados

---

## Filosofia del Paper

Este paper no es teorico. Cada concepto se acompana de codigo funcional
que el lector puede ejecutar. La progresion es:

  Neurona -> Red -> Transformer -> LLM -> Inferencia -> Agente -> Seguridad

Cada paso construye sobre el anterior. Al terminar, el lector entiende
exactamente que hay detras de un "agente de IA" y por que la seguridad
no se resuelve con un system prompt.

## Requisitos para ejecutar el codigo
- Python 3.10+
- PyTorch (para el transformer)
- numpy (para redes neuronales basicas)
- openai (para conectar a vLLM/cualquier API compatible)
- Acceso a un modelo con tool calling (Qwen, GPT, Claude, etc.)
- NO se necesita GPU para los ejemplos didacticos
