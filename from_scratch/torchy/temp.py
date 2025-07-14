from torch_ode_training import train_func
import numpy as np
import torch
import matplotlib.pyplot as plt

# PLOTTING SETTINGS
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})
colorcitos_gamma = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#583419', '#ffa11b', '#d1dc00'])
colorcitos_n = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#583419', '#ffa11b'])

ns = [10, 20, 40, 50, 60, 70]
gammas = [0.5, 1, 5, 10, 20, 50, 100]
n_input = 40
gamma = 10
n_validation = 80
L = 3
m = 10

xlims = [0, np.pi]
bc_type = [1, 1]
bcs = [0, 1]

# ODE RHS
f_torch = lambda x, y, y1st: 3*y1st - y + torch.cos(x)
f_np = lambda x, y, y1st: 3*y1st - y + np.cos(x)

losses_gamma = {}
iters_gamma = {}
for idx, gamma in enumerate(gammas):
    x, y_hat, sol, val_losses, iterations = train_func(n_input, n_validation, xlims, L, m, bc_type, bcs, f_torch, f_np,
                                                       gamma=gamma, loss_vs_iterations=True)
    losses_gamma[idx] = val_losses
    iters_gamma[idx] = iterations

losses_n = {}
iters_n = {}
for idx, n in enumerate(ns):
    x, y_hat, sol, val_losses, iterations = train_func(n, n_validation, xlims, L, m, bc_type, bcs, f_torch, f_np,
                                                       gamma=gamma, loss_vs_iterations=True)
    losses_n[idx] = val_losses
    iters_n[idx] = iterations

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
for i in range(len(losses_n)):
    ax[0].plot(iters_n[i], losses_n[i], color=next(colorcitos_n), label=f'N = {ns[i]}')
for i in range(len(losses_gamma)):
    ax[1].plot(iters_gamma[i], losses_gamma[i], color=next(colorcitos_gamma), label=f'$\gamma$ = {gammas[i]}')

ax[0].set_xlabel('Iterations')
ax[1].set_xlabel('Iterations')
ax[0].set_ylabel('Validation loss')
ax[1].set_ylabel('Validation loss')
ax[0].set_yscale('log')
ax[1].set_yscale('log')

ax[0].legend(fontsize=15)
ax[1].legend(fontsize=15)
plt.show()