#!/usr/bin/env python3
"""
perceptron.py -- Perceptron desde cero, sin librerias.
Parte 1 del paper "Hackers Artificiales o Sistemas Mal Disenados?"

Aprende la funcion AND logica y demuestra:
- Una neurona es multiplicacion + suma + activacion
- Los pesos se ajustan con cada error (aprendizaje)
- No hay nada magico. Son operaciones aritmeticas.
"""

import random


def perceptron(entradas, pesos, bias):
    """
    Una neurona: multiplica cada entrada por su peso,
    suma todo, agrega el bias, y aplica activacion escalon.
    """
    suma = sum(e * p for e, p in zip(entradas, pesos)) + bias
    return 1 if suma >= 0 else 0


def entrenar_perceptron(datos_entrenamiento, tasa_aprendizaje=0.1, epocas=100):
    """
    Entrena un perceptron ajustando los pesos con cada error.
    Regla de aprendizaje: peso_nuevo = peso_viejo + tasa * error * entrada
    """
    num_entradas = len(datos_entrenamiento[0][0])
    pesos = [random.uniform(-0.5, 0.5) for _ in range(num_entradas)]
    bias = random.uniform(-0.5, 0.5)

    print(f"Pesos iniciales: {[round(p, 4) for p in pesos]}")
    print(f"Bias inicial:    {round(bias, 4)}")
    print()

    for epoca in range(epocas):
        errores_totales = 0

        for entradas, esperado in datos_entrenamiento:
            prediccion = perceptron(entradas, pesos, bias)
            error = esperado - prediccion

            if error != 0:
                errores_totales += 1
                for i in range(len(pesos)):
                    pesos[i] += tasa_aprendizaje * error * entradas[i]
                bias += tasa_aprendizaje * error

        if errores_totales == 0:
            print(f"Convergio en epoca {epoca + 1}")
            break

    print(f"Pesos finales: {[round(p, 4) for p in pesos]}")
    print(f"Bias final:    {round(bias, 4)}")
    return pesos, bias


# ---- AND ----
datos_AND = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

print("=" * 50)
print("ENTRENANDO PERCEPTRON - Funcion AND")
print("=" * 50)

random.seed(42)
pesos, bias = entrenar_perceptron(datos_AND)

print()
print("Verificacion:")
for entradas, esperado in datos_AND:
    resultado = perceptron(entradas, pesos, bias)
    estado = "OK" if resultado == esperado else "ERROR"
    print(f"  {entradas[0]} AND {entradas[1]} = {resultado}  (esperado: {esperado})  [{estado}]")


# ---- OR ----
datos_OR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

print()
print("=" * 50)
print("ENTRENANDO PERCEPTRON - Funcion OR")
print("=" * 50)

random.seed(42)
pesos, bias = entrenar_perceptron(datos_OR)

print()
print("Verificacion:")
for entradas, esperado in datos_OR:
    resultado = perceptron(entradas, pesos, bias)
    estado = "OK" if resultado == esperado else "ERROR"
    print(f"  {entradas[0]} OR {entradas[1]} = {resultado}  (esperado: {esperado})  [{estado}]")

print()
print("NOTA: El perceptron puede aprender AND y OR pero NO puede aprender XOR.")
print("Para XOR necesitamos multiples capas (ver red_neuronal.py).")
