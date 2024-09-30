import numpy as np
import matplotlib.pyplot as plt

n_values = [10, 100, 1000]
errors = []


def f(x):
    return -4 * np.pi ** 2 * np.sin(2 * np.pi * x)


def phi_i(x, i, h):
    if (i - 1) * h <= x < i * h:
        return (x - (i - 1) * h) / h
    elif i * h <= x < (i + 1) * h:
        return ((i + 1) * h - x) / h
    else:
        return 0


def integrate_rhs(i, n):
    h = 1.0 / n
    x_vals = np.linspace(0, 1, n)
    integral = 0
    for j in range(1, len(x_vals)):
        x_left = x_vals[j - 1]
        x_right = x_vals[j]
        mid_x = (x_left + x_right) / 2
        integral += (x_right - x_left) * f(mid_x) * phi_i(mid_x, i, h)

    return integral


# Построение матрицы жесткости K и вектора правой части F
def build_system(n):
    h = 1.0 / n
    K = np.zeros((n - 1, n - 1))
    F = np.zeros(n - 1)

    # Матрица жесткости K
    for i in range(n - 1):
        if i > 0:
            K[i, i - 1] = 1 / h
        K[i, i] = -2 / h
        if i < n - 2:
            K[i, i + 1] = 1 / h

    # Вектор правой части F
    for i in range(1, n):
        F[i - 1] = integrate_rhs(i, n)

    return K, F


def galerkin_method(n):
    K, F = build_system(n)
    p = np.linalg.solve(K, F)
    p = np.concatenate(([0], p, [0]))
    return p


# Построение графика и вычисление ошибки
def plot_solution(n_values):

    for n in n_values:
        p_exact = []
        plt.figure(figsize=(10, 6))
        x_fem = np.linspace(0, 1, n+1)
        for x in x_fem:
            p_exact.append( np.sin(2 * np.pi * x))
        p_fem = galerkin_method(n)
        print(p_fem)
        print(p_exact)
        error = np.max(np.abs(p_fem - p_exact))
        errors.append(error)
        plt.plot(x_fem, p_fem, label=f'n={n}', marker='o', markersize=4)

    plt.plot(x_fem, p_exact, label='Аналитика', linestyle='--', color='black')

    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Сравнение')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Вывод ошибки
    for n, error in zip(n_values, errors):
        print(f"Ошибка для n={n}: {error}")

plot_solution(n_values)
