import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import ode_nn_armijo as nn

# Network parameters
m = 25      # Number of neurons in hidden layer

# Training parameters
N = 20      # Number of inputs
gamma = 0.9
tol = 5e-3
eps = 1e-4  # Perturbation for finite differences

# State the problem & boundary conditions
def f(x, y):
    # Returns function
    return -2*np.cosh(x)
# Boundary conditions
a = 0
b = 1
y_a = 0
y_b = 3

# Create input matrix X
x_vals = np.linspace(a, b, N+1)
X = np.zeros((2, N+1))
X[0, :] = np.ones(N+1)
X[1, :] = x_vals

# Initialise random weights and biases
W1 = np.random.rand(m, 2)
W2 = np.random.rand(1, m+1)

# Train the net
W1, W2 = nn.train(X, f, W1, W2, tol, N, m, y_a, y_b, eps, gamma)
y_hat, _, _ = nn.forward_propagation(X, W1, W2, N, m)
plt.scatter(x_vals, y_hat[0], color='green')

# Plot exact solution
def sys(x, y):
    f_vals = [f(xval, y[0]) for xval in x]
    return np.vstack((y[1], f_vals))

def bc(ya, yb):
    return np.array([ya[0] - y_a, yb[0] - y_b])

y_exact = np.zeros((2, x_vals.size))
sol_exact = solve_bvp(sys, bc, x_vals, y_exact)
plt.plot(x_vals, sol_exact.sol(x_vals)[0])

plt.show()