import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

h = 1 / 8
N = int(1 / h) + 1
re = 0.2 * h
rw = 0.001
pb_inj = 1
pb_prod = -1
alpha = 2 * np.pi / np.log(re / rw) * h**2
inj_i, inj_j = int(0.25 / h), int(0.25 / h)
prod_i, prod_j = int(0.75 / h), int(0.75 / h)

A = lil_matrix((N * N, N * N))
B = np.zeros(N * N)

def idx(i, j):
    return i * N + j

for i in range(N):
    for j in range(N):
        if i == 0:
            A[idx(i, j), idx(i, j)] = 1
            A[idx(i, j), idx(i + 1, j)] = -1
        elif i == N - 1:
            A[idx(i, j), idx(i, j)] = 1
            A[idx(i, j), idx(i - 1, j)] = -1
        elif j == 0:
            A[idx(i, j), idx(i, j)] = 1
            A[idx(i, j), idx(i, j+1)] = -1
        elif j == N - 1:
            A[idx(i, j), idx(i, j)] = 1
            A[idx(i, j), idx(i, j-1)] = -1
        else:
            A[idx(i, j), idx(i, j)] = -4
            A[idx(i, j), idx(i+1, j)] = 1
            A[idx(i, j), idx(i-1, j)] = 1
            A[idx(i, j), idx(i, j+1)] = 1
            A[idx(i, j), idx(i, j-1)] = 1


A[idx(inj_i, inj_j), idx(inj_i, inj_j)] -= alpha
A[idx(prod_i, prod_j), idx(prod_i, prod_j)] -= alpha
B[idx(inj_i, inj_j)] = alpha * pb_inj
B[idx(prod_i, prod_j)] = alpha * pb_prod
print(A)

P = spsolve(A, B)
P_matrix = P.reshape((N, N))
print(P_matrix)
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

plt.imshow(P_matrix)
plt.colorbar(label='Давленьице')
plt.title('Давленьице')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_yaxis()
plt.show()
