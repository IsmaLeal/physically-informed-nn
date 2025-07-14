import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import nn_versatile as nn

# Network parameters
m = 40      # Number of neurons in hidden layer

# Training parameters
N = 20      # Number of inputs
gamma = 0.1
tol = 1e-3
eps = 1e-3  # Perturbation for finite differences

# State the problem & boundary conditions
def f(x, y):
    # Returns function
    return -x**(x**(np.sin(x))) + np.cosh(x) #requires higher tolerance
# Boundary conditions
a = 0
b = 1
y_a = 1
y_b = -1
bc1_type = int(input(f'Choose the type of BC for x=a:\n'
                    '1. Dirichlet\n2. Von Neumann\n'
                    '3. Robin: '))
bc2_type = int(input(f'Choose the type of BC for x=b:\n'
                    '1. Dirichlet\n2. Von Neumann\n'
                    '3. Robin: '))

# Create input matrix X
x_vals = np.linspace(a, b, N+1)
X = np.zeros((2, N+1))
X[0, :] = np.ones(N+1)
X[1, :] = x_vals

# Initialise random weights and biases
W1 = np.random.rand(m, 2)
W2 = np.random.rand(1, m+1)
#W2[0, 0] = 0

# Train the net
W1, W2 = nn.train(X, f, W1, W2, tol, N, m, y_a, bc1_type, y_b, bc2_type, eps, gamma)
y_hat, _, _ = nn.forward_propagation(X, W1, W2, N, m)
plt.scatter(x_vals, y_hat[0], color='green')

# Plot exact solution
def sys(x, y):
    f_vals = [f(xval, y[0]) for xval in x]
    #f_vals = f(x, y[0])
    return np.vstack((y[1], f_vals))

def bc(ya, yb):
    if bc1_type == 1:
        bc_a = ya[0]
    elif bc1_type == 2:
        bc_a = ya[1]
    elif bc1_type == 3:
        bc_a = ya[0] + ya[1]

    if bc2_type == 1:
        bc_b = yb[0]
    elif bc2_type == 2:
        bc_b = yb[1]
    elif bc2_type == 3:
        bc_b = yb[0] + yb[1]

    return np.array([bc_a - y_a, bc_b - y_b])

y_exact = np.zeros((2, x_vals.size))
sol_exact = solve_bvp(sys, bc, x_vals, y_exact)
plt.plot(x_vals, sol_exact.sol(x_vals)[0])

plt.show()