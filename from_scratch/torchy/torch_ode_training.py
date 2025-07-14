import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_nn import OdeNN, ODE_training, ODE_loss
from scipy.integrate import solve_bvp
import time

# PLOTTING SETTINGS
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 20})

## NETWORK TRAINING
def train_func(n_inputs, n_validation, xlims, L, m, BC_type, BCs, f_torch, f_np,
               iterations=10000, gamma=10, loss_vs_iterations=False):
    ## TRAINING SET
    x_min = xlims[0]
    x_max = xlims[1]
    x_vals = torch.linspace(x_min, x_max, n_inputs).unsqueeze(1)
    x_np = x_vals.detach().numpy().flatten()

    # VALIDATION SET
    x_validation = torch.linspace(x_min, x_max, n_validation).unsqueeze(1)
    x_validation_np = x_validation.detach().numpy().flatten()
    x_validation = x_validation.clone().detach().requires_grad_(True)

    ts = []
    losses = []
    losses_val = []

    for i in range(1):
        net = OdeNN(1, L, m, 1)
        opt = torch.optim.Adam(net.parameters(), 1e-3)

        # Training
        start_time = time.time()
        if not loss_vs_iterations:
            loss, val_loss = ODE_training(net, x_vals, x_validation, ODE_loss, optimiser=opt, iterations=iterations,
                                      f=f_torch, bc=BCs, bc_type=BC_type, gamma=gamma, loss_vs_iterations=loss_vs_iterations)
        else:
            loss, val_loss, val_losses, iterations = ODE_training(net, x_vals, x_validation, ODE_loss, optimiser=opt, iterations=iterations,
                                      f=f_torch, bc=BCs, bc_type=BC_type, gamma=gamma, loss_vs_iterations=loss_vs_iterations)
        end_time = time.time()
        training_time = end_time - start_time

        # Forward pass
        y_hat = net(x_validation)

        # Performance
        losses.append(loss)
        losses_val.append(val_loss)
        ts.append(training_time)
    print(f'Overall performance\n\tFinal loss: {np.mean(losses)}')
    print(f'\tValidation loss: {np.mean(losses_val)}')
    print(f'\tTraining time: {np.mean(ts)}')
    ## ANALYTICAL SOL.
    def sys(x, y):
        u1, u2 = y
        f_vals = f_np(x, u1, u2)
        return np.vstack((u2, f_vals))
    def bc(ya, yb):
        if BC_type[0] == 1:
            bc_a = ya[0]
        elif BC_type[0] == 2:
            bc_a = ya[1]
        elif BC_type[0] == 3:
            bc_a = ya[0] + ya[1]

        if BC_type[1] == 1:
            bc_b = yb[0]
        elif BC_type[1] == 2:
            bc_b = yb[1]
        elif BC_type[1] == 3:
            bc_b = yb[0] + yb[1]

        return np.array([bc_a - BCs[0], bc_b - BCs[1]])
    # Solve
    y = np.zeros((2, x_validation_np.size))
    sol = solve_bvp(sys, bc, x_validation_np, y)

    if not loss_vs_iterations:
        return x_validation_np, y_hat.detach().numpy(), sol
    else:
        return x_validation_np, y_hat.detach().numpy(), sol, val_losses, iterations


if __name__ == '__main__':
    # Network architecture
    L = 3
    m = 10
    n_inputs = 30
    n_validation = 80

    # Boundary conditions
    xlims = [0, np.pi]
    BC_type = [1, 1]
    bcs = [0, 1]

    # ODE RHS
    f_torch = lambda x, y, y1st: 3 * y1st - y + torch.cos(x)
    f_np = lambda x, y, y1st: 3 * y1st - y + np.cos(x)

    x, y_hat, sol = train_func(n_inputs, n_validation, xlims, L, m, BC_type, bcs, f_torch, f_np)

    ## PLOT
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.scatter(x, y_hat, label='Network predictions $\hat{y}$', color='orange', s=10, zorder=2)
    ax.plot(x, sol.sol(x)[0], label='Numerical solution', color='blue', linewidth=2, zorder=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.legend()
    plt.show()
    a = 1