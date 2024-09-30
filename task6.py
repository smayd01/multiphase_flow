import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры задачи
L = 100  # длина области
T = 2  # конечное время
A = 1  # площадь поперечного сечения
q = 1  # поток воды
Swr = 0.2  # остаточная насыщенность воды
Sor = 0.15  # остаточная насыщенность нефти
mu_o = 50  # вязкость нефти
mu_w = 0.5  # вязкость воды
phi = 0.2
k_ro_star = 1  # максимальная проницаемость нефти
k_rw_star = 0.6  # максимальная проницаемость воды
No = 2  # параметр Брукса-Кори для нефти
Nw = 2  # параметр Брукса-Кори для воды
h = 1  # пространственный шаг
dt = 0.1  # временной шаг

# Пространственная сетка
Nx = int(L / h) + 1
x = np.linspace(0, L, Nx)

# Временная сетка
Nt = int(T / dt) + 1
t = np.linspace(0, T, Nt)

def effective_saturation_oil(Sw):
    return (Sw - Swr) / (1 - Sor - Swr)

def effective_saturation_water(So):
    return (So - Sor) / (1 - Sor - Swr)

def relative_permeability_oil(So):
    Se = effective_saturation_oil(So)
    return k_ro_star * Se**No

def relative_permeability_water(Sw):
    Se = effective_saturation_water(Sw)
    return k_rw_star * Se**Nw

def fractional_flow_oil(Sw):
    So = 1 - Sw
    lmbd_o = relative_permeability_oil(So) / mu_o
    lmbd_w = relative_permeability_water(Sw) / mu_w
    return lmbd_o/(lmbd_o + lmbd_w)

def fractional_flow_water(Sw):
    So = 1 - Sw
    lmbd_o = relative_permeability_oil(So) / mu_o
    lmbd_w = relative_permeability_water(Sw) / mu_w
    return lmbd_w/(lmbd_o + lmbd_w)

def F(Sw):
    return fractional_flow_water(Sw) / (fractional_flow_oil(Sw) + fractional_flow_water(Sw))

def numerical_dF(Sw, epsilon=1e-5):
    return (F(Sw + epsilon) - F(Sw - epsilon)) / (2 * epsilon)

max_iter = 10000
tol = 1e-10

def newton_method(S_old):
    S_new = S_old.copy()
    for _ in range(max_iter):
        G = np.zeros_like(S_new)
        J = np.zeros((Nx, Nx))

        # Внутренние узлы
        for i in range(1, Nx - 1):
            G[i] = S_new[i] - S_old[i] + ((dt * q) / (A * phi * 2 * h)) * (F(S_new[i+1]) - F(S_new[i-1]))
            J[i, i] = 1
            J[i, i - 1] = -((dt * q) / (A * phi * 2 * h)) * numerical_dF(S_new[i - 1])
            J[i, i + 1] = ((dt * q) / (A * phi * 2 * h)) * numerical_dF(S_new[i + 1])

        # Граничные условия
        G[0] = S_new[0] - S_old[0]
        J[0, 0] = 1

        # Для последней ячейки
        G[-1] = S_new[-1] - S_old[-1] + ((dt * q) / (A * phi * 2 * h)) * (F(S_new[-1]) - F(S_new[-1 - 1]))
        J[-1, -1] = 1 + ((dt * q) / (A * phi * 2 * h)) * numerical_dF(S_new[-1])
        J[-1, -2] = - ((dt * q) / (A * phi * 2 * h)) * numerical_dF(S_new[-2])

        delta_S = np.linalg.solve(J, -G)
        S_new += delta_S

        # Ограничение значений
        # S_new = np.clip(S_new, Swr, 1.0)

        if np.linalg.norm(delta_S, np.inf) < tol:
            break

    return S_new

# Начальные условия для насыщенности воды
S_w_init = np.ones(Nx) * Swr  # Инициализация насыщенности воды
S_all = np.zeros((Nt, Nx))    # Массив для хранения всех значений насыщенности на каждом временном шаге

# Начальное условие: полная насыщенность водой на входе
S_w_init[0] = 1

# Устанавливаем начальную насыщенность для временного шага n = 0
S_w = S_w_init.copy()
S_all[0, :] = S_w

# Цикл по времени
for n in range(1, Nt):
    S_w_new = newton_method(S_w)  # Метод Ньютона для обновления насыщенности воды
    S_all[n, :] = S_w_new         # Сохраняем результат для текущего временного шага
    S_w = S_w_new                 # Обновляем S_w для следующего временного шага

    # print(f'Step {n}: {S_w}')  # Выводим насыщенность на каждом шаге

Sw_front = 0.28322
def analytical(Sw, tau):
    numerator = -(12480000 * Sw ** 2 - 13104000 * Sw + 2121600)
    denominator = (595360000 * Sw ** 4 - 501664000 * Sw ** 3 +
                   166629600 * Sw ** 2 - 25679440 * Sw + 1560001)
    if denominator != 0:
        x = tau / 0.2 * (numerator / denominator)
        if x >= L:
            return None
        else:
            return x
    else:
        return None  # Если деление на ноль

# Задаем массив для значений Sw
Swater_left = np.arange(1, Sw_front, -0.0001)

x_left = []
valid_Swater_left = []

for Sw in Swater_left:
    result = analytical(Sw, T)
    if result is not None:
        x_left.append(result)
        valid_Swater_left.append(Sw)

x_left_max = max(x_left)

if x_left_max < L:
    x_right = np.arange(x_left_max, L, 0.01)
    Swater_right = np.full(len(x_right), 0.2)  # Фиксированное значение Sw = 0.2

    x_combined = np.concatenate((x_left, x_right))
    Sw_combined = np.concatenate((valid_Swater_left, Swater_right))
else:
    x_combined = x_left
    Sw_combined = valid_Swater_left

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_combined, Sw_combined, label='Аналитическое решение', linestyle='--', color='black')
plt.plot(x, S_all[-1, :], label='Численное решение', color='blue')
plt.xlabel('x')
plt.ylabel('Sw')
plt.legend()
plt.grid(True)
plt.title(f'Сравнение аналитического и численного решений на T = {T}')
plt.show()