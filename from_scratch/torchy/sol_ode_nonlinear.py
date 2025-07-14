import matplotlib.pyplot as plt
from torch_ode_training import train_func
import torch
import numpy as np

# Plotting settings
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

# Network architecture
L = 7
m = 40

# Boundary conditions
bc_types = [(1,1), (1,3), (2,2), (2,3)]
bcs = [0, 1]

# ODE RHS
f_torch = lambda x, y, y1st: torch.exp(-x)*torch.cos(y*x**2)/(1+x**2) + (y1st**2)*y + 10*torch.cos(6*x)
f_np = lambda x, y, y1st: np.exp(-x)*np.cos(y*x**2)/(1+x**2) + (y1st**2)*y + 10*np.cos(6*x)

xs = {}
y_hats = {}
ys = {}

for idx, bc_type in enumerate(bc_types):
    x, y_hat, sol = train_func(L, m, bc_type, bcs, f_torch, f_np)
    xs[idx] = x
    y_hats[idx] = y_hat
    ys[idx] = sol.sol(x)[0]


fig, a = plt.subplots(2, 2, figsize=(10,6))
plt.subplots_adjust(top=0.8, wspace=0.1)

a[0,0].scatter(xs[0], y_hats[0], label='Network predictions $\hat{y}$', color='orange', s=10, zorder=2)
a[0,0].plot(xs[0], ys[0], label='Numerical solution', color='blue', linewidth=2, zorder=1)
a[0,0].set_title('Dirichlet boundary conditions')
a[0,0].set_ylabel('$y$')
plt.setp(a[0,0].get_xticklabels(), visible=False, fontsize=12)
plt.setp(a[0,0].get_yticklabels(), fontsize=12)

a[0,1].scatter(xs[1], y_hats[1], color='orange', s=10, zorder=2)
a[0,1].plot(xs[1], ys[1], color='blue', linewidth=2, zorder=1)
a[0,1].set_title('Dirichlet-Robin boundary conditions')
plt.setp(a[0,1].get_xticklabels(), visible=False, fontsize=12)
plt.setp(a[0,1].get_yticklabels(), fontsize=12)

a[1,0].scatter(xs[2], y_hats[2], color='orange', s=10, zorder=2)
#a[1,0].plot(xs[2], ys[2], color='blue', linewidth=2, zorder=1)
a[1,0].set_title('Von Neumann boundary conditions')
a[1,0].set_ylabel('$y$')
a[1,0].set_xlabel('$x$')
plt.setp(a[1,0].get_xticklabels(), fontsize=12)
plt.setp(a[1,0].get_yticklabels(), fontsize=12)

a[1,1].scatter(xs[3], y_hats[3], color='orange', s=10, zorder=2)
a[1,1].plot(xs[3], ys[3], color='blue', linewidth=2, zorder=1)
a[1,1].set_title('Von Neumann-Robin boundary conditions')
a[1,1].set_xlabel('$x$')
plt.setp(a[1,1].get_xticklabels(), fontsize=12)
plt.setp(a[1,1].get_yticklabels(), fontsize=12)

fig.legend(loc='upper center')
plt.show()