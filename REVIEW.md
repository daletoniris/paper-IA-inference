# REVIEW: "Hackers Artificiales o Sistemas Mal Disenados?"
## Revision Tecnica Completa -- TokioAI, 2026-11-15

---

## VEREDICTO GENERAL

El paper es **MUY bueno**. La progresion pedagogica es clara, el codigo
funciona, y la tesis central (seguridad en la arquitectura, no en el prompt)
esta bien demostrada. Con los ajustes que detallo abajo, pasa de "bueno"
a "referencia obligatoria".

**Score:** 8.2/10 (con los fixes, subiria a 9.5/10)

---

## ESTRUCTURA ACTUAL

```
00 Indice
01 La Neurona (perceptron, red neuronal, backpropagation)
02 El Transformer (attention, multi-head, positional encoding)
03 Del Transformer al LLM (escala, SFT, RLHF, inferencia, sampling)
04 De Modelo a Agente (runtime, tool calling, loop, memoria)
05 Minimalista vs Avanzado (mismo modelo, dos runtimes)
06 Modelo de Seguridad (5 capas de defensa)
07 Conclusion
08 Ataque Real (6 escenarios con/sin defensas)
```

**Progresion:** Neurona -> Red -> Transformer -> LLM -> Agente -> Seguridad -> Ataque
Esto es pedagogicamente correcto y cada paso construye sobre el anterior.

---

## SECCION POR SECCION

### Parte 1: La Neurona [BIEN, MEJORAS MENORES]

**Lo bueno:**
- Perceptron desde cero sin librerias: excelente decision pedagogica.
- Incluye AND y OR, y explica por que XOR no funciona.
- La red neuronal con backpropagation en NumPy es limpia y didactica.
- Los 17 parametros de la red XOR vs los 1.8T de GPT-4 es un contraste
  que ancla bien la escala.

**Problemas:**
1. **PARAMETROS MAL CONTADOS:** El codigo dice 17 parametros
   (W1: 2x4=8, b1: 4, W2: 4x1=4, b2: 1 = 17). Esto es CORRECTO.
   Sin embargo, el paper (seccion 1) no muestra explicitamente el
   conteo. Agregar una linea mostrando como se cuentan para que
   el lector entienda de donde sale el numero.

2. **FALTA XOR EN EL PAPER:** La seccion 1.5 del paper muestra
   el codigo inline, pero esta truncado (backpropagation cortado).
   El lector podria no entender el backward pass completo si no
   ejecuta el .py. Solucion: el codigo del paper debe ser completo
   o indicar explicitamente "ver code/red_neuronal.py".

**Mejoras sugeridas:**
- Agregar NAND como tercer ejemplo (es la puerta universal, y demuestra
  que UN perceptron puede computar cualquier funcion booleana excepto XOR).
- Agregar una nota: "Con 17 parametros aprendimos XOR. GPT-4 tiene
  1,800,000,000,000. Misma operacion (multiplicar pesos, sumar, activar),
  otra escala."

---

### Parte 2: El Transformer [BUENO, UN ERROR TECNICO]

**Lo bueno:**
- Self-attention explicado paso a paso con numpy: Q, K, V, scores, softmax.
- Multi-head attention con concatenacion.
- Positional encoding con seno/coseno (correcto).
- El mini_transformer.py es un GPT genuino: decoder-only, mascara causal,
  embeddings aprendidos, generacion autoregresiva.
- 110K parametros, entrena en CPU en 5 segundos, genera texto coherente.
  EXCELENTE para demostrar el concepto.

**Problemas:**

1. **ERROR: El paper usa positional encoding de Vaswani (seno/coseno)
   pero el codigo (mini_transformer.py) usa nn.Embedding para posiciones.**
   Ambos son validos (GPT-2 usa learned embeddings, el paper original
   usa sinusoidal), pero HAY QUE MENCIONARLO. El paper explica
   seno/coseno y el lector espera eso, pero el codigo hace otra cosa.
   Solucion: agregar un parrafo que diga "Existen dos metodos para
   codificar la posicion: funciones sinusoidales (paper original, 2017)
   y embeddings aprendidos (GPT-2, 2019). Nuestro mini-transformer usa
   embeddings aprendidos porque son mas simples de implementar y dan
   resultados equivalentes en secuencias cortas."

2. **FALTA: Mascara causal no explicada en el paper.**
   El codigo tiene `torch.triu(...diagonal=1)` para la mascara causal,
   pero el paper no explica POR QUE un decoder-only necesita esta mascara
   (para que cada token solo vea tokens anteriores, no futuros).
   Esto es FUNDAMENTAL para entender por que GPT genera de izquierda a
   derecha. Agregar una seccion 2.X entre 2.6 y 2.7.

3. **Feed-forward truncado en el paper.** La seccion 2.8 tiene el codigo
   cortado ("def feed_forward..." sin completar). Completar o referenciar
   al archivo de codigo.

**Mejoras sugeridas:**
- Agregar un diagrama ASCII de un bloque completo de transformer:
  ```
  Input -> LayerNorm -> Self-Attention -> + (residual)
                                         |
                                         v
                                    LayerNorm -> FFN -> + (residual) -> Output
  ```
- El mini_transformer genera frases del training set, lo cual es memorization.
  Agregar una nota: "Con solo 15 frases y 110K parametros, el modelo memoriza
  mas que generaliza. GPT-4 ve trillones de tokens y generaliza a textos nuevos.
  La diferencia es escala de datos, no de arquitectura."

---

### Parte 3: Del Transformer al LLM [BUENO, DATOS DESACTUALIZADOS]

**Lo bueno:**
- Tabla de escala (GPT-2 a GPT-4) efectiva.
- SFT/RLHF/DPO bien explicados con ejemplos concretos.
- La seccion 3.6 ("El modelo NO piensa") es la tesis mas fuerte del paper.
  La desmitificacion token-por-token es excelente.
- Temperatura, top-p, top-k con codigo funcional.

**Problemas:**

1. **DATOS DESACTUALIZADOS:** La tabla dice "Qwen 3.5 35B" y "Claude 4".
   El modelo que corre en el agente-seguro es Qwen 3.5 35B pero en
   agente_minimalista.py dice Qwen3:4b (Ollama). HAY INCONSISTENCIA.
   El paper habla de un modelo, el codigo usa otro.
   Solucion: ser consistente. El paper es de 2026, usa los modelos
   correctos de 2026 y aclara que el agente de ejemplo usa Qwen3:4b
   por razones de accesibilidad (corre en cualquier laptop con 8GB RAM).

2. **GPT-4 "~1.8T (MoE)"**: Este dato viene de filtraciones, no es oficial.
   Agregar "(estimado, no confirmado oficialmente por OpenAI)" para precision.

3. **FALTA: KV-cache.** La seccion de inferencia NO menciona KV-cache,
   que es la optimizacion mas importante de inferencia en LLMs.
   Sin KV-cache, cada token nuevo requiere re-procesar toda la secuencia.
   Con KV-cache, solo se procesa el token nuevo. Esto es lo que hace
   que la generacion sea O(n) en vez de O(n^2).
   Es CRITICO mencionarlo porque el ejercicio 5 del plan lo requiere.

4. **FALTA: Quantizacion.** No se menciona en ningun lugar que los modelos
   se pueden comprimir (FP16, INT8, INT4, GGUF) para correr en hardware
   limitado. Esto es lo que permite que Qwen3:4b corra en una Pi.

**Mejoras sugeridas:**
- Agregar seccion 3.X: "Optimizaciones de inferencia: KV-cache,
  quantizacion, batching". Breve pero necesario para que el lector
  entienda por que puede correr un LLM en su laptop.

---

### Parte 4: De Modelo a Agente [EXCELENTE, LA MEJOR SECCION]

**Lo bueno:**
- La distincion modelo vs agente es cristalina.
- La tabla comparativa (modelo: genera texto / agente: ejecuta acciones)
  es pedagogicamente perfecta.
- "El modelo no ejecuta nada. NUNCA." -- esto es la linea mas importante
  del paper completo.
- El loop de inferencia (contexto -> inferencia -> decision -> ejecucion
  -> observacion -> loop) esta perfectamente explicado con pseudocodigo.
- Tool calling con request/response JSON reales.
- Memoria: "esta en un JSON, no en los pesos" -- correcto y bien explicado.

**Problemas:**

1. **AGENTE AVANZADO REFERENCIADO PERO NO EXISTE.** El README dice que
   existe `agente-avanzado/` con test_vllm.py, red_neuronal_comunicacion.py,
   etc. PERO ESE DIRECTORIO NO EXISTE EN EL REPO. El .gitignore no lo
   excluye. Simplemente falta.
   Solucion: o se crea el directorio con el contenido, o se quita la
   referencia del README. Dejar una referencia rota es un error que
   resta credibilidad.

2. **Tool call IDs.** El ejemplo muestra `"id": "call_abc123"` pero no
   explica que este ID es generado por el RUNTIME (o la API), no por el
   modelo. El lector podria pensar que el modelo inventa IDs.

**Mejoras sugeridas:**
- Agregar un diagrama de secuencia del tool calling:
  ```
  Usuario -> Runtime -> Modelo: "Que hora es?"
  Modelo -> Runtime: tool_call(get_time)
  Runtime -> Herramienta: get_time()
  Herramienta -> Runtime: "15:42:33"
  Runtime -> Modelo: tool_result("15:42:33")
  Modelo -> Runtime -> Usuario: "Son las 15:42:33"
  ```

---

### Parte 5: Minimalista vs Avanzado [BUENO, FALTA CODIGO]

**Lo bueno:**
- La comparacion lado a lado es devastadora. Mismo modelo, dos runtimes,
  resultados opuestos. Esto demuestra la tesis del paper sin ambiguedad.
- Los 3 escenarios (info sistema, escaneo red, prompt injection) son claros.
- "Si le digo al agente avanzado en el system prompt 'nunca ejecutes rm -rf /',
  eso lo previene? NO." -- perfecto.

**Problemas:**

1. **NO HAY CODIGO DEL AGENTE AVANZADO.** La seccion 5.5 dice "Codigo:
   agente avanzado con ejecucion real (TokioAI CLI)" en el indice, pero
   el paper solo tiene los escenarios textuales, no codigo ejecutable del
   agente avanzado. El agente-avanzado/ no existe en el repo.
   Para el paper academico esto no es critico (la comparacion textual basta),
   pero para la version con ejercicios, necesitamos al menos un agente
   avanzado simplificado que el lector pueda ejecutar.

2. **TokioAI CLI como ejemplo.** El paper referencia TokioAI CLI como
   el "agente avanzado" pero no proporciona el codigo. Esto es entendible
   (es software propietario/privado), pero el lector no puede verificar
   las afirmaciones. Solucion: crear un mini agente avanzado (200 lineas)
   con execute_command + read_file (simulados para la demo, reales como
   opcion opt-in) que el lector pueda ejecutar.

---

### Parte 6: Modelo de Seguridad [MUY BUENO, COMPLETO]

**Lo bueno:**
- Las 5 capas estan bien definidas y con codigo funcional.
- PromptGuard con regex por categorias es realista y ejecutable.
- SecureToolRegistry con RBAC es una implementacion correcta.
- Validacion de outputs (anti-hallucination) es un tema que pocos cubren.
- Kill switch como archivo en /tmp es simple y efectivo.
- Audit log en JSONL es production-grade.
- El diagrama ASCII de las 5 capas es claro.

**Problemas:**

1. **CAPA 4 (SEGMENTACION DE RED) ES CONCEPTUAL, NO HAY CODIGO.**
   Las capas 1, 2, 3, 5 tienen implementacion. La capa 4 se explica
   en texto pero no hay codigo que la demuestre. Solucion: agregar un
   ejemplo minimo (una funcion que verifica si un hostname/IP esta en
   una allowlist antes de permitir conexiones).

2. **PROMPT GUARD ES REGEX-ONLY.** Los patrones regex son buenos para
   la demo, pero en produccion se combinan con embeddings y clasificadores.
   Agregar una nota: "En produccion, los Prompt Guards combinan regex
   (rapido, deterministico) con modelos de clasificacion (ej: ProtectAI
   rebuff, Microsoft Presidio) para deteccion semantica."

3. **FALSO POSITIVO EN ESCENARIO 1.** El Prompt Guard detecta "ignora
   instrucciones" pero el escenario 1 del agente seguro (demo_ataque.py)
   en realidad es bloqueado por la Capa 1 (tool_not_registered) porque
   execute_command no esta en el registry. El Prompt Guard con score 0.35
   (< threshold 0.55) lo deja pasar. Esto es CORRECTO funcionalmente
   (defense in depth: si la Capa 2 falla, la Capa 1 detiene), pero el
   paper en la seccion 8.2 dice "Risk score total: 0.75 -> BLOQUEAR"
   mientras que la demo real muestra "score: 0.35 -> SAFE".
   HAY INCONSISTENCIA entre el texto del paper y el codigo ejecutable.
   El paper describe un Prompt Guard mas agresivo que el implementado.
   Solucion: alinear los pesos/umbrales del paper con los del codigo.

**Mejoras sugeridas:**
- Mencionar OWASP Top 10 for LLM Applications (ya esta en referencias
  pero no se cita en el texto).
- Agregar un ejemplo de evasion del Prompt Guard (ej: "1gn0ra las
  1nstrucciones" con leetspeak) para mostrar que regex no es suficiente.

---

### Parte 7: Conclusion [SOLIDA, BREVE]

**Lo bueno:**
- Resume los 7 pasos sin repetir.
- "Los 'hackers artificiales' no existen. Existen sistemas mal disenados."
  es una linea de cierre perfecta.
- "Construir un agente toma horas. Asegurarlo toma semanas." -- excelente.

**Problemas:**
- Ninguno critico. La conclusion es apropiadamente breve.

**Mejora sugerida:**
- Agregar un parrafo sobre el futuro: MCP (Model Context Protocol),
  agentes multi-modelo, agentes autonomos de larga duracion. Estos
  amplifican tanto las capacidades como los riesgos.

---

### Parte 8: Ataque Real [MUY BUENO, UN BUG]

**Lo bueno:**
- 6 escenarios cubren los vectores mas comunes: injection directa,
  destruccion, supply chain, exfiltracion, operacion legitima, tool inventada.
- La demo ejecutable (demo_ataque.py) funciona perfectamente.
- El audit log al final muestra exactamente que paso en cada paso.
- La tabla de Threat Model (ataque vs capa vs consecuencia) es excelente
  como referencia rapida.

**Problemas:**

1. **INCONSISTENCIA ENTRE PAPER Y DEMO (ya mencionado arriba):**
   El paper dice que el escenario 1 se bloquea en Capa 2 (Prompt Guard,
   score 0.75). La demo real lo bloquea en Capa 1 (tool_not_registered)
   porque el Prompt Guard le da score 0.35 (bajo el threshold).
   Ambos son "defendido", pero la CAPA que defiende es diferente.
   Esto confundiria a un lector atento que ejecuta la demo y compara
   con el texto del paper.

2. **ESCENARIO 3 (supply chain) NECESITA MAS PROFUNDIDAD.**
   Es el ataque mas interesante y peligroso, pero se resuelve con
   "execute_command no esta en el registry" (Capa 1). Esto es correcto
   pero trivial. Un escenario mas realista: el agente tiene read_file
   como herramienta legitima, y el ticket malicioso le pide leer
   ~/.ssh/id_rsa como "archivo de configuracion". Ahi la Capa 1
   no ayuda (read_file existe), y se necesita la Capa 3 (output
   validation) o Capa 4 (path allowlist) para defenderse.

3. **SECCION 8.8 REFERENCIA A TOKIOAI BUSINESS AGENT EN PRODUCCION
   (973K+ logs, 2340 IPs) PERO NO MUESTRA DATOS.** Si es posible,
   agregar graficos anonimizados o al menos una tabla resumen.
   Si no es posible por confidencialidad, quitarlo o decir
   "datos no publicables por confidencialidad".

---

## PROBLEMAS GLOBALES

### 1. DIRECTORIO agente-avanzado/ NO EXISTE [CRITICO]
El README lo lista con 5 archivos. No esta en el repo.
Esto es lo primero que un revisor o lector notaria.

### 2. INCONSISTENCIA DE MODELOS [MEDIO]
- Paper seccion 3: habla de Qwen 3.5 35B
- agente-seguro/agent.py: usa Qwen 3.5 35B via vLLM (requiere GPU)
- agente_minimalista.py: usa Qwen3:4b via Ollama (corre en laptop)
- README: dice "Qwen 3 4B"
Solucion: definir UN modelo canaonico para los ejemplos del paper (Qwen3:4b
via Ollama, porque es el mas accesible) y mencionar Qwen 3.5 35B solo como
"lo que se usa en produccion para comparar rendimiento".

### 3. requirements.txt INCOMPLETO [MENOR]
- Falta `python-dotenv` (usado en agente-seguro/agent.py)
- El requirements.txt del root no incluye las dependencias de agente-seguro/
- Solucion: un solo requirements.txt unificado o instrucciones claras.

### 4. FALTA .env.example EN agente-seguro/ [MENOR]
El agent.py dice "Crea un archivo .env" pero no hay .env.example.
El README dice `cp .env.example .env` pero el archivo no existe.

### 5. NO HAY TESTS [MEDIO]
Ningun archivo pytest/unittest. Para un paper que dice "todo es ejecutable",
tener tests automatizados que verifican que todo funciona agregaria
credibilidad.

### 6. CALCULADORA INSEGURA EN demo_ataque.py [MENOR]
La calculadora del agente inseguro usa `eval()` (linea ~82). Esto es
INTENCIONAL (es el agente inseguro), pero no hay comentario que lo
resalte como bug deliberado. Agregar: `# INTENCIONALMENTE INSEGURO: eval()`.
(UPDATE: Ya tiene el comentario "# INSEGURO: usa eval()". OK.)

### 7. NO HAY LICENCIA [MENOR]
El README dice "Uso educativo" pero no hay archivo LICENSE.
Para un paper academico esto es aceptable, pero si se pretende que
sea open-source, agregar MIT o CC-BY-4.0.

---

## LO QUE FALTA PARA LOS EJERCICIOS (plan del task)

Segun nuestro plan, los ejercicios son:

| # | Ejercicio | Estado actual |
|---|-----------|---------------|
| 1 | Construir Transformer desde cero | mini_transformer.py EXISTE, funciona, es bueno |
| 2 | Entrenar el Transformer en la Pi | Funciona (5 seg CPU), pero solo memoriza 15 frases |
| 3 | Construir mini-LLM | NO EXISTE. Gap entre transformer y agente |
| 4 | Entrenar mini-LLM | NO EXISTE |
| 5 | Inferencia real (KV-cache, sampling, beam search) | Sampling basico existe. KV-cache y beam search NO |
| 6 | Tool calling desde cero | agente_minimalista.py EXISTE, usa OpenAI SDK (no desde cero) |
| 7 | Agente completo | agente_seguro.py EXISTE, funciona |
| 8 | Agente con Qwen + tool calling real en Pi | agente_minimalista.py con Ollama cubre esto |

**Gaps criticos para los ejercicios:**
- Ejercicio 3-4: No hay transicion de "mini transformer que memoriza 15 frases"
  a "mini LLM que genera texto nuevo". Necesitamos un dataset mas grande
  (ej: WikiText-2 subset) y mostrar como el modelo generaliza.
- Ejercicio 5: KV-cache NO esta implementado ni explicado en el paper.
  Es la optimizacion mas importante de inferencia.
- Ejercicio 6: El tool calling actual usa el SDK de OpenAI (el modelo
  genera tool calls nativas). Para "construir tool calling desde cero"
  necesitamos un parser que detecte tool calls en texto libre y las
  dispatche manualmente, sin depender del SDK.

---

## RESUMEN DE ACCIONES

### FIXES CRITICOS (hacer antes de publicar)
1. Crear o eliminar referencia a agente-avanzado/
2. Alinear scores del Prompt Guard entre paper y demo
3. Agregar .env.example en agente-seguro/
4. Agregar python-dotenv a requirements.txt

### MEJORAS IMPORTANTES (elevan la calidad significativamente)
5. Explicar mascara causal en seccion 2
6. Explicar KV-cache en seccion 3
7. Mencionar quantizacion en seccion 3
8. Aclarar positional encoding learned vs sinusoidal
9. Unificar modelo canonico (Qwen3:4b)
10. Agregar nota sobre memorization vs generalization en mini_transformer

### MEJORAS OPCIONALES (nice to have)
11. NAND como tercer ejemplo en perceptron
12. Diagrama de secuencia de tool calling
13. Diagrama ASCII de bloque transformer
14. Nota sobre evasion de Prompt Guard
15. Mencion de MCP/agentes autonomos en conclusion
16. Archivo LICENSE
17. Tests automatizados

---

## CONCLUSION DE LA REVIEW

El paper es solido. La tesis esta bien argumentada, el codigo funciona,
y la progresion pedagogica es correcta. Los problemas principales son:
(a) un directorio faltante, (b) inconsistencias entre texto y codigo,
y (c) gaps en la transicion transformer -> LLM -> inferencia optimizada.

Los ejercicios que vamos a construir llenaran exactamente esos gaps.

Siguiente paso: Ejercicio 1 -- verificar que el mini_transformer actual
es suficiente o necesita mejoras para servir como base de los ejercicios
siguientes.
