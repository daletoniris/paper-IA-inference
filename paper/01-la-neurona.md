# Parte 1: La Neurona -- Donde Empieza Todo

## 1.1 Que es una neurona artificial

Una neurona artificial es una funcion matematica que recibe numeros,
los multiplica por pesos, los suma, y pasa el resultado por una
funcion de activacion.

Eso es todo. No hay magia. No hay conciencia. Es multiplicacion y suma.

```
entrada_1 ---[peso_1]---\
                          \
entrada_2 ---[peso_2]----->  SUMA + bias  --->  activacion  --->  salida
                          /
entrada_3 ---[peso_3]---/
```

La formula:

```
salida = activacion(peso_1 * entrada_1 + peso_2 * entrada_2 + ... + bias)
```

Los **pesos** determinan cuanto influye cada entrada en la salida.
El **bias** es un offset que permite ajustar el umbral de activacion.
La **funcion de activacion** introduce no-linealidad (sin ella, toda
la red seria equivalente a una unica multiplicacion de matrices).

## 1.2 Perceptron: la unidad minima de decision

El perceptron es la neurona artificial mas simple. Fue inventado en 1958
por Frank Rosenblatt. Recibe entradas, las multiplica por pesos, suma
todo, y devuelve 1 o 0.

Puede aprender a clasificar datos linealmente separables (como una linea
que divide dos grupos de puntos en un plano).

## 1.3 Codigo: perceptron desde cero

```python
#!/usr/bin/env python3
"""
perceptron.py -- Perceptron desde cero, sin librerias.
Aprende la funcion AND logica.
"""

import random


def perceptron(entradas, pesos, bias):
    """
    Una neurona: multiplica cada entrada por su peso,
    suma todo, agrega el bias, y aplica activacion escalon.

    Si la suma >= 0 -> devuelve 1
    Si la suma <  0 -> devuelve 0
    """
    suma = sum(e * p for e, p in zip(entradas, pesos)) + bias

    # Funcion de activacion: escalon (step function)
    if suma >= 0:
        return 1
    else:
        return 0


def entrenar_perceptron(datos_entrenamiento, tasa_aprendizaje=0.1, epocas=100):
    """
    Entrena un perceptron ajustando los pesos con cada error.

    datos_entrenamiento: lista de tuplas (entradas, salida_esperada)
    tasa_aprendizaje: cuanto se ajustan los pesos en cada correccion
    epocas: cuantas veces se recorre todo el dataset
    """

    # Inicializar pesos aleatorios pequenos
    num_entradas = len(datos_entrenamiento[0][0])
    pesos = [random.uniform(-0.5, 0.5) for _ in range(num_entradas)]
    bias = random.uniform(-0.5, 0.5)

    print(f"Pesos iniciales: {[round(p, 4) for p in pesos]}")
    print(f"Bias inicial:    {round(bias, 4)}")
    print()

    for epoca in range(epocas):
        errores_totales = 0

        for entradas, esperado in datos_entrenamiento:
            # Paso 1: Calcular salida actual
            prediccion = perceptron(entradas, pesos, bias)

            # Paso 2: Calcular error
            error = esperado - prediccion

            if error != 0:
                errores_totales += 1

                # Paso 3: Ajustar pesos y bias
                # Esta es la regla de aprendizaje del perceptron:
                # peso_nuevo = peso_viejo + tasa * error * entrada
                for i in range(len(pesos)):
                    pesos[i] += tasa_aprendizaje * error * entradas[i]

                bias += tasa_aprendizaje * error

        # Si no hubo errores en toda la epoca, ya convergio
        if errores_totales == 0:
            print(f"Convergio en epoca {epoca + 1}")
            break

    print(f"Pesos finales: {[round(p, 4) for p in pesos]}")
    print(f"Bias final:    {round(bias, 4)}")

    return pesos, bias


# ---- Dataset: funcion AND ----
# Dos entradas binarias, salida = AND logico
# AND solo devuelve 1 cuando AMBAS entradas son 1.
datos_AND = [
    ([0, 0], 0),   # 0 AND 0 = 0
    ([0, 1], 0),   # 0 AND 1 = 0
    ([1, 0], 0),   # 1 AND 0 = 0
    ([1, 1], 1),   # 1 AND 1 = 1
]

print("=" * 50)
print("ENTRENANDO PERCEPTRON - Funcion AND")
print("=" * 50)

pesos, bias = entrenar_perceptron(datos_AND)

print()
print("Verificacion:")
for entradas, esperado in datos_AND:
    resultado = perceptron(entradas, pesos, bias)
    estado = "OK" if resultado == esperado else "ERROR"
    print(f"  {entradas[0]} AND {entradas[1]} = {resultado}  (esperado: {esperado})  [{estado}]")
```

**Que demuestra este codigo:**
- Una neurona es multiplicacion + suma + activacion.
- Los pesos se ajustan con cada error (aprendizaje).
- Despues de pocas epocas, el perceptron "aprende" la funcion AND.
- No hay nada magico. Son operaciones aritmeticas.

**Limitacion:** un perceptron solo puede aprender funciones linealmente separables.
No puede aprender XOR (donde los puntos no se pueden separar con una linea recta).
Para eso necesitamos multiples capas.


## 1.4 Redes neuronales: capas de neuronas conectadas

Una red neuronal es simplemente multiples capas de neuronas donde la salida
de una capa se convierte en la entrada de la siguiente.

```
CAPA DE ENTRADA        CAPA OCULTA         CAPA DE SALIDA
  (datos)            (representacion)       (prediccion)

  entrada_1 ----\     /--- neurona_1 ---\     /--- salida
                 \   /                    \   /
  entrada_2 ------X-------neurona_2 ------X------ salida
                 /   \                    /   \
  entrada_3 ----/     \--- neurona_3 ---/     \--- salida
```

- **Capa de entrada**: recibe los datos crudos (numeros).
- **Capa oculta**: transforma los datos. Puede haber 1 o muchas.
- **Capa de salida**: produce la prediccion final.

Cada conexion tiene un peso. Cada neurona tiene un bias.
El entrenamiento ajusta TODOS estos pesos y biases para minimizar el error.

## 1.5 Codigo: red neuronal feedforward con backpropagation

```python
#!/usr/bin/env python3
"""
red_neuronal.py -- Red neuronal de 2 capas desde cero con NumPy.
Aprende la funcion XOR (que un perceptron simple no puede).
"""

import numpy as np

# Semilla para reproducibilidad
np.random.seed(42)


# ---- Funciones de activacion ----

def sigmoid(x):
    """
    Sigmoid: convierte cualquier numero a un valor entre 0 y 1.
    Es la funcion de activacion clasica de redes neuronales.
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivada(x):
    """
    Derivada de sigmoid. Se usa en backpropagation para
    calcular cuanto ajustar cada peso.
    """
    s = sigmoid(x)
    return s * (1 - s)


# ---- Dataset: funcion XOR ----
# XOR devuelve 1 cuando las entradas son diferentes.
# Un perceptron simple NO puede aprender esto.
# Necesitamos al menos una capa oculta.

X = np.array([
    [0, 0],   # 0 XOR 0 = 0
    [0, 1],   # 0 XOR 1 = 1
    [1, 0],   # 1 XOR 0 = 1
    [1, 1],   # 1 XOR 1 = 0
])

Y = np.array([
    [0],
    [1],
    [1],
    [0],
])


# ---- Arquitectura de la red ----
# 2 entradas -> 4 neuronas ocultas -> 1 salida

neuronas_entrada = 2
neuronas_ocultas = 4
neuronas_salida  = 1

# Inicializar pesos aleatorios (distribucion normal pequena)
# Estos pesos son EXACTAMENTE lo que el entrenamiento va a ajustar.
pesos_entrada_oculta = np.random.randn(neuronas_entrada, neuronas_ocultas) * 0.5
bias_oculta = np.zeros((1, neuronas_ocultas))

pesos_oculta_salida = np.random.randn(neuronas_ocultas, neuronas_salida) * 0.5
bias_salida = np.zeros((1, neuronas_salida))

# Hiperparametros
tasa_aprendizaje = 1.0
epocas = 10000

print("=" * 50)
print("ENTRENANDO RED NEURONAL - Funcion XOR")
print(f"Arquitectura: {neuronas_entrada} -> {neuronas_ocultas} -> {neuronas_salida}")
print(f"Tasa de aprendizaje: {tasa_aprendizaje}")
print(f"Epocas: {epocas}")
print("=" * 50)


# ---- Entrenamiento con backpropagation ----
for epoca in range(epocas):

    # ======== FORWARD PASS (propagacion hacia adelante) ========
    # Calcular la salida de la red paso a paso.

    # Capa oculta: entrada * pesos + bias, luego activacion
    z_oculta = X @ pesos_entrada_oculta + bias_oculta   # Combinacion lineal
    a_oculta = sigmoid(z_oculta)                          # Activacion

    # Capa de salida: salida_oculta * pesos + bias, luego activacion
    z_salida = a_oculta @ pesos_oculta_salida + bias_salida
    a_salida = sigmoid(z_salida)

    # ======== CALCULAR ERROR ========
    # Error = diferencia entre la prediccion y el valor real
    error = Y - a_salida
    loss = np.mean(error ** 2)  # Mean Squared Error

    # ======== BACKWARD PASS (backpropagation) ========
    # Calcular cuanto debe cambiar cada peso para reducir el error.
    # Esto usa la regla de la cadena del calculo diferencial.

    # Gradiente de la capa de salida
    delta_salida = error * sigmoid_derivada(z_salida)

    # Gradiente de la capa oculta (propagado desde la salida)
    error_oculta = delta_salida @ pesos_oculta_salida.T
    delta_oculta = error_oculta * sigmoid_derivada(z_oculta)

    # ======== ACTUALIZAR PESOS ========
    # peso_nuevo = peso_viejo + tasa * gradiente
    pesos_oculta_salida += a_oculta.T @ delta_salida * tasa_aprendizaje
    bias_salida += np.sum(delta_salida, axis=0, keepdims=True) * tasa_aprendizaje

    pesos_entrada_oculta += X.T @ delta_oculta * tasa_aprendizaje
    bias_oculta += np.sum(delta_oculta, axis=0, keepdims=True) * tasa_aprendizaje

    # Mostrar progreso cada 2000 epocas
    if (epoca + 1) % 2000 == 0:
        print(f"  Epoca {epoca + 1:>5}: loss = {loss:.6f}")

# ---- Resultado final ----
print()
print("Verificacion:")
for i in range(len(X)):
    prediccion = a_salida[i][0]
    redondeado = round(prediccion)
    esperado = Y[i][0]
    estado = "OK" if redondeado == esperado else "ERROR"
    print(f"  {X[i][0]} XOR {X[i][1]} = {prediccion:.4f} "
          f"(redondeado: {redondeado}, esperado: {esperado})  [{estado}]")

print()
print(f"Parametros totales: {pesos_entrada_oculta.size + bias_oculta.size + pesos_oculta_salida.size + bias_salida.size}")
print()
print("NOTA: Los pesos finales SON el conocimiento de la red.")
print("Si guardas los pesos, guardas lo que la red 'aprendio'.")
print("Si cargas los pesos en otra maquina, obtenes la misma red entrenada.")
print("Esto es exactamente lo que pasa con GPT, Claude, Qwen, etc.")
print("La diferencia: ellos tienen miles de millones de pesos en vez de 16.")
```

**Que demuestra este codigo:**
- Forward pass: los datos fluyen por la red multiplicandose por pesos.
- Backpropagation: el error se propaga hacia atras para ajustar cada peso.
- Los pesos SON el conocimiento. No hay "memoria magica".
- Con solo 16 parametros, la red aprende XOR.
- GPT-4 tiene ~1.8 trillones de parametros. Mismo principio, otra escala.


## 1.6 Por que esto importa: los pesos SON el conocimiento

**Concepto clave para todo lo que viene:**

Cuando decimos que un modelo "sabe" algo, lo que realmente decimos es que
los valores numericos de sus pesos producen la salida correcta para esa
entrada. No hay una "base de datos de conocimiento" separada. No hay un
"cerebro" que "piensa".

- **Entrenamiento** = ajustar pesos para minimizar errores en un dataset.
- **Inferencia** = usar esos pesos (fijos, sin cambiar) para procesar nuevas entradas.
- **El modelo no cambia cuando le haces preguntas.** Los pesos son los mismos.

Esto es fundamental para entender agentes de IA:
- El modelo no "aprende" de tu conversacion (los pesos no cambian).
- La "memoria" de un agente es contexto en el prompt, no aprendizaje real.
- La "inteligencia" del agente depende del runtime que lo rodea, no solo del modelo.

---

*En la siguiente seccion, vamos a escalar esto: de 16 parametros a la
arquitectura Transformer, que es la base de todos los LLMs actuales.*
