#!/usr/bin/env python3
"""
red_neuronal.py -- Red neuronal de 2 capas desde cero con NumPy.
Parte 1 del paper "Hackers Artificiales o Sistemas Mal Disenados?"

Aprende la funcion XOR que un perceptron simple no puede.
Demuestra forward pass, backpropagation, y por que los pesos SON el conocimiento.
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    """Sigmoid: cualquier numero -> valor entre 0 y 1."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivada(x):
    """Derivada de sigmoid, usada en backpropagation."""
    s = sigmoid(x)
    return s * (1 - s)


# ---- Dataset: XOR ----
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# ---- Arquitectura: 2 -> 4 -> 1 ----
n_input, n_hidden, n_output = 2, 4, 1

W1 = np.random.randn(n_input, n_hidden) * 0.5
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * 0.5
b2 = np.zeros((1, n_output))

lr = 1.0
epochs = 10000

print("=" * 50)
print("ENTRENANDO RED NEURONAL - Funcion XOR")
print(f"Arquitectura: {n_input} -> {n_hidden} -> {n_output}")
print(f"Parametros: {W1.size + b1.size + W2.size + b2.size}")
print("=" * 50)

for epoch in range(epochs):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Loss
    error = Y - a2
    loss = np.mean(error ** 2)

    # Backward
    d2 = error * sigmoid_derivada(z2)
    d1 = (d2 @ W2.T) * sigmoid_derivada(z1)

    # Update
    W2 += a1.T @ d2 * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X.T @ d1 * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    if (epoch + 1) % 2000 == 0:
        print(f"  Epoca {epoch + 1:>5}: loss = {loss:.6f}")

print()
print("Verificacion:")
for i in range(len(X)):
    pred = a2[i][0]
    r = round(pred)
    e = Y[i][0]
    print(f"  {X[i][0]} XOR {X[i][1]} = {pred:.4f} (redondeado: {r}, esperado: {e})  [{'OK' if r == e else 'ERROR'}]")

print()
print(f"Total parametros: {W1.size + b1.size + W2.size + b2.size}")
print("Los pesos finales SON el conocimiento de la red.")
print("GPT-4 tiene ~1.8T parametros. Mismo principio, otra escala.")
