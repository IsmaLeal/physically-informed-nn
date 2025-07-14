import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import nn_fullynonlinear as nn
import time

# PLOTTING SETTINGS
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 14})


def train_func(N, xlims, m, bc_type, bcs, f, tol=5e-3, eps=1e-3, gamma=10):
    a, b = xlims[0], xlims[1]
    y_a, y_b = bcs[0], bcs[1]

    # Create input matrix X
    x_vals = np.linspace(a, b, N+1)
    X = np.zeros((2, N+1))
    X[0, :] = np.ones(N+1)
    X[1, :] = x_vals

    # Initialise random weights and biases
    W1 = np.random.rand(m, 2)
    W2 = np.random.rand(1, m+1)
    W2[0, 0] = 0

    # Train the net
    start_time = time.time()
    W1, W2, loss = nn.train(X, f, W1, W2, tol, N, m, y_a, bc_type[0], y_b, bc_type[1], eps, gamma)
    end_time = time.time()
    training_time = end_time - start_time

    # Forward pass
    y_hat, _, _ = nn.forward_propagation(X, W1, W2, N, m)

    # Stats information
    print(f'Overall performance\n\tFinal loss: {loss}')
    print(f'\tTraining time: {training_time}')

    # Analytical sol.
    def sys(x, y):
        u1, u2 = y
        f_vals = f(x, u1, u2)
        return np.vstack((u2, f_vals))

    def bc(ya, yb):
        if bc_type[0] == 1:
            bc_a = ya[0]
        elif bc_type[0] == 2:
            bc_a = ya[1]
        elif bc_type[0] == 3:
            bc_a = ya[0] + ya[1]

        if bc_type[1] == 1:
            bc_b = yb[0]
        elif bc_type[1] == 2:
            bc_b = yb[1]
        elif bc_type[1] == 3:
            bc_b = yb[0] + yb[1]

        return np.array([bc_a - y_a, bc_b - y_b])

    y_initial_guess = np.zeros((2, x_vals.size))
    sol_exact = solve_bvp(sys, bc, x_vals, y_initial_guess)

    return x_vals, y_hat, sol_exact


if __name__ == '__main__':
    # Network parameters
    m = 40  # Number of neurons in hidden layer

    # Training parameters
    N = 40  # Number of inputs
    gamma = 10
    tol = 1e-2
    eps = 1e-3  # Perturbation for finite differences

    # Domain
    xlims = [0, np.pi]
    bc_types = [(1,1), (1,2), (2,1), (2,2)]
    bcs = [0, 1]


    # State the problem & boundary conditions
    def f(x, y, y_1st):
        # Returns function
        return -2*x**2 + y # 3 * y_1st - y + np.cos(x)


    xs = {}
    y_hats = {}
    ys = {}

    for idx, bc_type in enumerate(bc_types):
        x, y_hat, sol = train_func(N, xlims, m, bc_type, bcs, f, tol=1e-2)
        xs[idx] = x
        y_hats[idx] = y_hat
        ys[idx] = sol.sol(x)[0]

    fig, a = plt.subplots(2, 2, figsize=(10,6))
    plt.subplots_adjust(top=0.8, wspace=0.1)

    a[0, 0].scatter(xs[0], y_hats[0], label='Network predictions $\hat{y}$', color='orange', s=10, zorder=2)
    a[0, 0].plot(xs[0], ys[0], label='Numerical solution', color='blue', linewidth=2, zorder=1)
    a[0, 0].set_title('Dirichlet boundary conditions')
    a[0, 0].set_ylabel('$y$')
    plt.setp(a[0, 0].get_xticklabels(), visible=False, fontsize=12)
    plt.setp(a[0, 0].get_yticklabels(), fontsize=12)

    a[0, 1].scatter(xs[1], y_hats[1], color='orange', s=10, zorder=2)
    a[0, 1].plot(xs[1], ys[1], color='blue', linewidth=2, zorder=1)
    a[0, 1].set_title('Dirichlet-Von Neumann boundary conditions')
    plt.setp(a[0, 1].get_xticklabels(), visible=False, fontsize=12)
    plt.setp(a[0, 1].get_yticklabels(), fontsize=12)

    a[1, 0].scatter(xs[2], y_hats[2], color='orange', s=10, zorder=2)
    a[1, 0].plot(xs[2], ys[2], color='blue', linewidth=2, zorder=1)
    a[1, 0].set_title('Von Neumann-Dirichlet boundary conditions')
    a[1, 0].set_ylabel('$y$')
    a[1, 0].set_xlabel('$x$')
    plt.setp(a[1, 0].get_xticklabels(), fontsize=12)
    plt.setp(a[1, 0].get_yticklabels(), fontsize=12)

    a[1, 1].scatter(xs[3], y_hats[3], color='orange', s=10, zorder=2)
    a[1, 1].plot(xs[3], ys[3], color='blue', linewidth=2, zorder=1)
    a[1, 1].set_title('Von Neumann boundary conditions')
    a[1, 1].set_xlabel('$x$')
    plt.setp(a[1, 1].get_xticklabels(), fontsize=12)
    plt.setp(a[1, 1].get_yticklabels(), fontsize=12)

    fig.legend(loc='upper center')
    plt.tight_layout()
    plt.show()