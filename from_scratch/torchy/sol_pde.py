import matplotlib.pyplot as plt
from torch_pde_training import train_func
import torch
import numpy as np

# Plotting settings
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

# Network architecture
L = 6
m = 40
n_inputs = 80
n_validation = 100

# Boundary conditions
xlims = [0, 2*np.pi]
ylims = [0, np.pi]
bcs = [0, 1]

# ODE RHS
f_torch = lambda u, x, y: -torch.sin(x)*torch.sin(y) - torch.cos(u)
f_np = lambda u, x, y: -np.sin(x)*np.sin(y) - np.cos(u)

# Anonymous function returning 0
zero_anon = lambda x, y: 0
neg_one_anon = lambda x, y: -1
one_anon = lambda x, y: 1
dirichlet_bc = lambda x, y: torch.exp(-y) * torch.sin(3*y)
sin_bc = lambda x, y: torch.sin(3*x)

xs = {}
ys = {}
u_hats = {}

for i in range(4):
    if i == 0: # Dirichlet with one wavy side
        bc_type = [1, 1, 1, 1]
        bcs = [zero_anon, zero_anon, zero_anon, dirichlet_bc]
    elif i == 1:
        bc_type = [3, 1, 1, 1]  # Dirichlet-Von Neumann
        bcs = [zero_anon, zero_anon, zero_anon, dirichlet_bc]
    elif i == 2:
        bc_type = [3, 3, 2, 2]  # All Von Neumann
        bcs = [zero_anon, neg_one_anon, zero_anon, zero_anon]
    elif i == 3:
        bc_type = [1, 4, 1, 1]  # Dirichlet-Robin
        bcs = [sin_bc, sin_bc, zero_anon, zero_anon]

    x, y, u_hat = train_func(n_inputs, n_validation, xlims, ylims, L, m, bc_type, bcs, f_torch, f_np,
                               iterations=5000, gamma=10)
    xs[i] = x
    ys[i] = y
    u_hats[i] = u_hat


fig, a = plt.subplots(2, 2, figsize=(10,6), subplot_kw={'projection': '3d'})

a[0,0].plot_surface(xs[0], ys[0], u_hats[0])
a[0,0].set_title('Dirichlet boundary conditions')
a[0,0].set_zlabel('$\hat{u}$')
a[0,0].set_xlabel('$x$')
a[0,0].set_ylabel('$y$')

a[0,1].plot_surface(xs[1], ys[1], u_hats[1])
a[0,1].set_title('Dirichlet-Von Neumann boundary conditions')
a[0,1].set_zlabel('$\hat{u}$')
a[0,1].set_xlabel('$x$')
a[0,1].set_ylabel('$y$')

a[1,0].plot_surface(xs[2], ys[2], u_hats[2])
a[1,0].set_title('Von Neumann boundary conditions')
a[1,0].set_zlabel('$\hat{u}$')
a[1,0].set_xlabel('$x$')
a[1,0].set_ylabel('$y$')

a[1,1].plot_surface(xs[3], ys[3], u_hats[3])
a[1,1].set_title('Dirichlet-Robin boundary conditions')
a[1,1].set_zlabel('$\hat{u}$')
a[1,1].set_xlabel('$x$')
a[1,1].set_ylabel('$y$')

plt.tight_layout()
plt.show()