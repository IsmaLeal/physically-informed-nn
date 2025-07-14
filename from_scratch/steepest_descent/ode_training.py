from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import numpy as np
import ode_nn

# Network parameters
m = 30       # Number of neurons in hidden layer

# Training parameters
N = 20      # Number of inputs
gamma = 0.6
learning_rate = 0.06
epochs = 10000
eps = 1e-3  # Perturbation for finite differences

# State the problem & boundary conditions
def f(x, y):
    # Returns function
    return -2 * x**2 + y
# Boundary conditions
a = 0
b = np.pi
y_a = 0
y_b = 1

# Create input matrix X
x_vals = np.linspace(a, b, N+1)
X = np.zeros((2, N+1))
X[0, :] = np.ones(N+1)
X[1, :] = x_vals

# Initialise random weights and biases
W1 = np.random.rand(m, 2)
W2 = np.random.rand(1, m+1)

# Train the net
W1, W2 = ode_nn.train(X, f, W1, W2, epochs, learning_rate, N, m, ya, yb, eps, gamma)
y_hat, _, _ = ode_nn.forward_propagation(X, W1, W2, N, m)
plt.scatter(x_vals, y_hat[0], color='green')

# Plot exact solution
def sys(x, y):
    f_vals = f(x, y[0])
    return np.vstack((y[1], f_vals))

def bc(ya, yb):
    return np.array([ya[0]-0, yb[0]-1])


x = np.linspace(0, np.pi, 300)
y_a = np.zeros((2, x.size))

sol = solve_bvp(sys, bc, x, y_a)

plt.plot(x, sol.sol(x)[0])
plt.grid()
plt.show()
