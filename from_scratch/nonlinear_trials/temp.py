from training_fullynonlinear import train_func
import numpy as np
import matplotlib.pyplot as plt


# Training parameters
m = 20
N = 40  # Number of inputs
gamma = 10
tol = 1e-2
eps = 1e-3  # Perturbation for finite differences

# Domain
xlims = [0, np.pi]
bc_type = [1, 2]
bcs = [0, 1]


# State the problem & boundary conditions
def f(x, y, y_1st):
    # Returns function
    return -2*x**2 + y #3 * y_1st - y + np.cos(x)
    #return x*y

x, y_hat, sol = train_func(N, xlims, m, bc_type, bcs, f, tol=1e-1)

fig, a = plt.subplots(1, 1, figsize=(10,6))

a.scatter(x, y_hat, label='Network predictions $\hat{y}$', color='orange', s=10, zorder=2)
a.plot(x, sol.sol(x)[0], label='Numerical solution', color='blue', linewidth=2, zorder=1)
a.set_title('Dirichlet boundary conditions')
a.set_ylabel('$y$')
plt.setp(a.get_xticklabels(), visible=False, fontsize=12)
plt.setp(a.get_yticklabels(), fontsize=12)

a.legend()
plt.tight_layout()
plt.show()