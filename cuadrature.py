#!/usr/bin/env python3
"""
cuadrature.py

Implementación del método de cuadratura gaussiana para resolver la integral:
    ∫₀^π sin(x²) dx

Incluye funciones auxiliares para escalado de intervalo y obtención de pesos/puntos.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

def gauss_legendre(n):
    """
    Obtiene los puntos de colocación y pesos para la cuadratura gaussiana
    con polinomios de Legendre.

    Parameters:
        n (int): Número de puntos de cuadratura.

    Returns:
        tuple: Dos arreglos numpy, uno con los puntos y otro con los pesos.

    Example:
        >>> x, w = gauss_legendre(3)
        >>> len(x)
        3
    """
    return leggauss(n)

def scale_interval(a, b, x):
    """
    Escala los puntos de colocación desde el intervalo [-1, 1] al intervalo [a, b].

    Parameters:
        a (float): Límite inferior del intervalo original.
        b (float): Límite superior del intervalo original.
        x (np.ndarray): Puntos en [-1, 1].

    Returns:
        np.ndarray: Puntos escalados al intervalo [a, b].

    Example:
        >>> scale_interval(0, 2, np.array([-1, 0, 1]))
        array([0., 1., 2.])
    """
    return 0.5 * (b - a) * x + 0.5 * (b + a)

def gaussian_quadrature(f, a, b, n):
    """
    Calcula una integral definida utilizando cuadratura gaussiana.

    Parameters:
        f (function): Función a integrar.
        a (float): Límite inferior de integración.
        b (float): Límite superior de integración.
        n (int): Número de puntos de colocación.

    Returns:
        float: Aproximación numérica de la integral.

    Example:
        >>> gaussian_quadrature(lambda x: x**2, 0, 1, 3)
        0.333...
    """
    x, w = gauss_legendre(n)
    x_scaled = scale_interval(a, b, x)
    return 0.5 * (b - a) * np.sum(w * f(x_scaled))

def main():
    """
    Ejecuta el cálculo de la integral y genera un gráfico de resultados.
    """
    f = lambda x: np.sin(x**2)
    a, b = 0, np.pi
    ns = range(2, 21)
    results = [gaussian_quadrature(f, a, b, n) for n in ns]

    plt.figure()
    plt.plot(ns, results, marker='o')
    plt.xlabel("Número de puntos N")
    plt.ylabel("Valor aproximado de la integral")
    plt.title("Aproximación de ∫₀^π sin(x²) dx con cuadratura gaussiana")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

