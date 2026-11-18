# Parte 7: Conclusion

## 7.1 No existen hackers artificiales

A lo largo de este paper recorrimos el camino completo:

1. **Neurona artificial** -- multiplicacion y suma. No hay magia.
2. **Red neuronal** -- capas de neuronas que ajustan pesos con backpropagation.
3. **Transformer** -- self-attention permite procesar secuencias completas en paralelo.
4. **LLM** -- un transformer gigante entrenado con trillones de tokens.
5. **Inferencia** -- prediccion del siguiente token, uno a la vez.
6. **Agente** -- un modelo + un runtime + herramientas + memoria.
7. **Seguridad** -- 5 capas de defensa en la arquitectura, no en el prompt.

En ningun paso aparecio "inteligencia", "conciencia", ni "voluntad".
Un LLM es una funcion matematica que calcula probabilidades.
Un agente es un programa que usa esa funcion para decidir acciones.

**Los "hackers artificiales" que aparecen en los titulares no existen.**

Lo que existe son:
- Sistemas con runtimes mal disenados que permiten acciones peligrosas.
- Agentes sin validacion de inputs expuestos a prompt injection.
- Agentes sin validacion de outputs que alucinan datos criticos.
- Infraestructura sin segmentacion donde el agente tiene acceso a todo.
- Sistemas sin observabilidad donde nadie ve lo que el agente hace.

El problema nunca fue el modelo. El problema siempre fue la arquitectura.


## 7.2 La seguridad esta en la arquitectura

Lo demostramos con codigo funcional:

- **Mismo modelo** (Qwen 3.5 35B) con dos runtimes diferentes
  produce dos agentes radicalmente distintos en capacidad y riesgo.
- **El agente minimalista** no puede hacer dano porque su runtime
  solo expone 3 herramientas inofensivas.
- **El agente avanzado** puede hacer cualquier cosa porque su runtime
  expone 12+ herramientas con acceso total.

La diferencia NO es el prompt. Es el TOOL_REGISTRY.
Es la validacion de argumentos. Es el firewall. Es el logging.

```
Prompt injection no es un bug del modelo.
Es un bug del sistema que confia en texto para hacer seguridad.
```


## 7.3 Cualquiera puede construir un agente. No cualquiera puede asegurarlo.

Con 300 lineas de Python y un modelo open-source, cualquiera puede
construir un agente funcional. Las librerias lo hacen trivial.

Lo que NO es trivial:
- Decidir que herramientas exponer y cuales no.
- Validar cada input antes de que el modelo lo procese.
- Verificar cada output antes de que el usuario lo reciba.
- Aislar al agente en una red controlada.
- Monitorear cada accion y tener un kill switch funcional.
- Anticipar que el modelo va a ser manipulado y disenar para eso.

**Construir un agente toma horas. Asegurarlo toma semanas.**

Y eso esta bien. Porque la seguridad no es un feature.
Es la diferencia entre un sistema que sirve y uno que destruye.


## Nota final

Todo el codigo de este paper es funcional y ejecutable.
El lector puede:

1. Correr el perceptron y la red neuronal para ver como aprenden los pesos.
2. Entrenar el mini-transformer en CPU y ver como genera texto.
3. Conectar el agente minimalista a cualquier modelo con tool calling.
4. Implementar las 5 capas de seguridad en su propio agente.

No hace falta creer. Hace falta ejecutar, leer el codigo, y entender.
La IA no es magia. Es matematica con buena ingenieria alrededor.

---

**Daniel Dieser**
Lead Incident Response & CyberSecurity
2026

---

## Referencias

1. Vaswani, A. et al. "Attention Is All You Need" (2017). arXiv:1706.03762
2. Radford, A. et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
3. Brown, T. et al. "Language Models are Few-Shot Learners" (GPT-3, 2020). arXiv:2005.14165
4. Ouyang, L. et al. "Training language models to follow instructions with human feedback" (InstructGPT/RLHF, 2022). arXiv:2203.02155
5. Rafailov, R. et al. "Direct Preference Optimization" (DPO, 2023). arXiv:2305.18290
6. Greshake, K. et al. "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection" (2023). arXiv:2302.12173
7. Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022). arXiv:2210.03629
8. Schick, T. et al. "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023). arXiv:2302.04761
9. OWASP Top 10 for LLM Applications (2025). https://owasp.org/www-project-top-10-for-large-language-model-applications/
10. Rosenblatt, F. "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)
