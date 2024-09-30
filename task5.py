import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# параметры сетки
L = 10000
h = 1000
N = int(L / h) + 1
t = 10
dt = 1

# параметры пласта
bw = 1
phi = 0.2
ct = 10e-6
k = 50
muw = 1

# параметры скважин
re = 0.2 * h
rw = 0.25
pb_prod = 800
q_inj = 10e-5
q_prod = 2 * np.pi / np.log(re / rw)
inj_i, inj_j = int(5000 / h), int(5000 / h)
prod_i, prod_j = int(9000 / h), int(9000 / h)

alpha = k * dt / (muw * phi * ct)
alpha_rhs = bw * dt / (phi * ct)

P_init = np.ones(N * N) * 1000

def idx(i, j):
    return i * N + j

# временной массив
t_array = np.arange(0, t + dt, dt)
P_solution = np.zeros((N * N, len(t_array)))

A = lil_matrix((N * N, N * N))
for i in range(N):
    for j in range(N):
        if i == 0:  # верхняя граница
            if j == 0:
                A[idx(i, j), idx(i, j)] = 1
                A[idx(i, j), idx(i + 1, j)] = -1
            elif j == N-1:
                A[idx(i, j), idx(i, j)] = 1
            elif 0 < j < N - 1:
                # A[idx(i, j), idx(i, j)] = 3 / h**2 * alpha + 1
                # A[idx(i, j), idx(i + 1, j)] = -alpha / h**2
                # A[idx(i, j), idx(i, j + 1)] = -alpha / h**2
                # A[idx(i, j), idx(i, j - 1)] = -alpha / h ** 2
                A[idx(i, j), idx(i, j)] = 1
                A[idx(i, j), idx(i + 1, j)] = -1
        elif i == N - 1:  # нижняя граница
            if j == 0:
                A[idx(i, j), idx(i, j)] = 1
                A[idx(i, j), idx(i - 1, j)] = -1
            elif j == N - 1:
                A[idx(i, j), idx(i, j)] = 1
            elif 0 < j < N - 1:
                # A[idx(i, j), idx(i, j)] = 3 / h**2 * alpha + 1
                # A[idx(i, j), idx(i - 1, j)] = -alpha / h**2
                # A[idx(i, j), idx(i, j + 1)] = -alpha / h ** 2
                # A[idx(i, j), idx(i, j - 1)] = -alpha / h**2
                A[idx(i, j), idx(i, j)] = 1
                A[idx(i, j), idx(i - 1, j)] = -1
        elif j == 0:  # левая граница
            if 0 < i < N - 1:
                # A[idx(i, j), idx(i, j)] = 3 / h**2 * alpha + 1
                # A[idx(i, j), idx(i + 1, j)] = -alpha / h**2
                # A[idx(i, j), idx(i - 1, j)] = -alpha / h ** 2
                # A[idx(i, j), idx(i, j + 1)] = -alpha / h ** 2
                A[idx(i, j), idx(i, j)] = 1
                A[idx(i, j), idx(i + 1, j)] = -1
        elif j == N - 1:  # правая граница - Дирихле p = 2000
            A[idx(i, j), idx(i, j)] = 1
        else:  # внутренняя область
            A[idx(i, j), idx(i, j)] = 4 / h**2 * alpha + 1
            A[idx(i, j), idx(i + 1, j)] = -alpha / h**2
            A[idx(i, j), idx(i - 1, j)] = -alpha / h**2
            A[idx(i, j), idx(i, j + 1)] = -alpha / h**2
            A[idx(i, j), idx(i, j - 1)] = -alpha / h**2

A[idx(prod_i, prod_j), idx(prod_i, prod_j)] += q_prod * alpha_rhs
print(A)
A = A.tocsc()
for tau_idx, tau in enumerate(t_array):
    if tau_idx == 0:
        B = P_init
        B[idx(inj_i, inj_j)] += q_inj * alpha_rhs
        B[idx(prod_i, prod_j)] += q_prod * pb_prod * alpha_rhs
        for i in range(N):
            for j in range(N):
                if i == 0 or j == 0 or i == N - 1:
                    B[idx(i, j)] = 0
                elif j == N - 1:
                    B[idx(i, j)] = 2000
        P = spsolve(A, B)
        P_solution[:, tau_idx] = P
        # print(B)
    else:
        B = P_solution[:, tau_idx - 1].copy()
        B[idx(inj_i, inj_j)] += q_inj * alpha_rhs
        B[idx(prod_i, prod_j)] += q_prod * pb_prod * alpha_rhs
        for i in range(N):
            for j in range(N):
                if (i == 0 and j == 0) or (i == N-1 and j == 0):
                    B[idx(i, j)] = 0
                elif j == N - 1:
                    B[idx(i, j)] = 2000
        # print(B)
        P = spsolve(A, B)
        # print(P)
        P_solution[:, tau_idx] = P

# визуализация результатов
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# # создание файлов VTK и анимации
grid = pv.ImageData()
grid.dimensions = (N+1, N+1, 1)
grid.spacing = (h, h, 1)

output_dir = 'vtk_output'
os.makedirs(output_dir, exist_ok=True)

for tau_idx, tau in enumerate(t_array):
    pressure_data = P_solution[:, tau_idx].reshape((N, N))
    print(pressure_data)
    grid.cell_data['Pressure'] = pressure_data.flatten()

    # сохраняем каждый временной шаг как файл VTK
    filename = f'{output_dir}/pressure_step_{tau_idx:03d}.vtk'
    grid.save(filename)
