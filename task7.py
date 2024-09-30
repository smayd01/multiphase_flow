import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Параметры задачи
L = 10000  # Длина области
T_end = 10  # Время моделирования
phi_m = 0.2  # Пористость матрицы
phi_f = 0.01  # Пористость трещины
k_m = 0.01  # Проницаемость матрицы
k_f = 50  # Проницаемость трещины
c_t = 10 * 1e-6  # Коэффициент сжимаемости
mu = 1  # Вязкость
lambda_ = 10e-6  # Коэффициент взаимодействия

# Шаги по пространству и времени
dx = 100  # Шаг по пространству
dt = 1  # Шаг по времени
Nx = int(L / dx) + 1  # Число точек по пространству
Nt = int(T_end / dt) + 1  # Число шагов по времени

# Начальные условия
p_m = np.ones(Nx) * 1000  # Давление в матрице
p_f = np.ones(Nx) * 1000  # Давление в трещине

# Граничные условия
p_left = 500  # Граничное условие слева
p_m[0] = p_left
p_f[0] = p_left


def jacobian(Nx):
    J = np.zeros((2 * Nx, 2 * Nx))
    for i in range(1, Nx - 1):
        # Для F1
        J[i, i] = phi_m * c_t / dt + 2 * k_m / (mu * dx ** 2) + lambda_ * k_m / mu
        J[i, i - 1] = -k_m / (mu * dx ** 2)
        J[i, i + 1] = -k_m / (mu * dx ** 2)
        J[i, i + Nx] = -lambda_ * k_m / mu

        # Для F2
        J[i + Nx, i + Nx] = phi_f * c_t / dt + 2 * k_f / (mu * dx ** 2) + lambda_ * k_f / mu
        J[i + Nx, i + Nx - 1] = -k_f / (mu * dx ** 2)
        J[i + Nx, i + Nx + 1] = -k_f / (mu * dx ** 2)
        J[i + Nx, i] = -lambda_ * k_f / mu

    J[0, 0] = 1

    J[Nx, Nx] = 1

    # J[-1 - Nx, -1 - Nx] = phi_m * c_t / dt + 2 * k_m / (mu * dx ** 2) + lambda_ * k_m / mu + k_m / (mu * dx ** 2)
    # J[-1 - Nx, -2 - Nx] = -k_m / (mu * dx ** 2)
    # J[-1 - Nx, -1] = -lambda_ * k_m / mu
    J[-1 - Nx, -1 - Nx] = 1/dx
    J[-1 - Nx, -2 - Nx] = -1/dx

    # J[-1, -1] = phi_f * c_t / dt + 2 * k_f / (mu * dx ** 2) + lambda_ * k_f / mu + k_f / (mu * dx ** 2)
    # J[-1, -2] = -k_f / (mu * dx ** 2)
    # J[-1, -1 - Nx] = -lambda_ * k_f / mu
    J[-1, -1] = 1/dx
    J[-1, -2] = -1/dx

    return J


# Итерации по времени
tolerance = 1e-3
max_iter = 1000
for t in range(Nt):
    # Сохраняем старые значения давлений
    p_m_old = p_m.copy()
    p_f_old = p_f.copy()
    error = np.max(abs(p_m_old - p_f_old))
    print(error, t)

    for n in range(max_iter):
        F = np.zeros(2 * Nx)
        # Составляем систему F1 и F2 для всех внутренних точек
        for i in range(1, Nx - 1):
            F[i] = phi_m * c_t * (p_m[i] - p_m_old[i]) / dt - k_m / mu * (p_m[i+1] - 2 * p_m[i] + p_m[i-1]) / dx ** 2 - k_m / mu * lambda_ * (p_f[i] - p_m[i])
            F[i + Nx] = phi_f * c_t * (p_f[i] - p_f_old[i]) / dt - k_f / mu * (p_f[i+1] - 2 * p_f[i] + p_f[i-1]) / dx ** 2 - k_f / mu * lambda_ * (p_m[i] - p_f[i])
        F[0] = p_m[0] - p_m_old[0]
        F[Nx] = p_f[0] - p_f_old[0]
        # F[-1 - Nx - 1] = phi_m * c_t * (p_m[-1] - p_m_old[-1]) / dt - k_m / mu * (- p_m[-1] + p_m[-2]) / dx ** 2 - k_m / mu * lambda_ * (p_f[-1] - p_m[-1])
        # F[-1] = phi_f * c_t * (p_f[-1] - p_f_old[-1]) / dt - k_f / mu * (- p_f[-1] + p_f[-2]) / dx ** 2 - k_f / mu * lambda_ * (p_m[-1] - p_f[-1])
        F[-1 - Nx - 1] = (p_m[-1] - p_m[-2]) / dx
        F[-1] = (p_f[-1] - p_f[-2]) / dx

        # Якобиан
        J = jacobian(Nx)

        delta_p = np.linalg.solve(J, -F)

        alpha = 1
        # Обновляем давления
        p_m += alpha * delta_p[0:Nx]
        p_f += alpha * delta_p[Nx:]

        # Проверяем условие остановки
        if np.linalg.norm(delta_p) < tolerance:
            break

    else:
        print(f"Не удалось достичь сходимости на шаге времени {t + 1}")


# Построение результатов
x = np.linspace(0, L, Nx)
plt.plot(x, p_m, label='Давление в матрице')
plt.plot(x, p_f, label='Давление в трещине')
plt.xlabel('x')
plt.ylabel('p')
plt.title('Давление')
plt.legend()
plt.show()
49+