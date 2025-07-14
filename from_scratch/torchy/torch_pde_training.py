import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_nn import PdeNN, PDE_training, PDE_loss
import time

# PLOTTING SETTINGS
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 14})

def train_func(n_inputs, n_validation, xlims, ylims, L, m, bc_type, bcs, f_torch, f_np,
               iterations=1000, gamma=10):
    x_min, x_max = xlims[0], xlims[1]
    y_min, y_max = ylims[0], ylims[1]
    ## TRAINING SET
    x_vals = torch.linspace(x_min, x_max, n_inputs)
    y_vals = torch.linspace(y_min, y_max, n_inputs)
    xx, yy = torch.meshgrid(x_vals, y_vals, indexing='xy')
    [xx, yy] = [xx.unsqueeze(2), yy.unsqueeze(2)]
    # VALIDATION SET
    x_validation_vals = torch.linspace(x_min, x_max, n_validation)
    y_validation_vals = torch.linspace(y_min, y_max, n_validation)
    xx_val, yy_val = torch.meshgrid(x_validation_vals, y_validation_vals, indexing='xy')
    [xx_val, yy_val] = [xx_val.unsqueeze(2), yy_val.unsqueeze(2)]

    net = PdeNN(2, L, m, 1)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    # Training
    start_time = time.time()
    loss, val_loss = PDE_training(net, xx, yy, xx_val, yy_val, PDE_loss, opt, iterations, f_torch,
                                  bcs, bc_type, gamma)
    end_time = time.time()
    training_time = end_time - start_time

    # Forward pass
    with torch.no_grad():
        u_hat = net(xx_val, yy_val).squeeze(2)
    print(f'\tLoss; {loss}\n\tValidation loss: {val_loss}\n\tTraining time: {training_time}')
    return xx_val.squeeze(2).detach().numpy(), yy_val.squeeze(2).detach().numpy(), u_hat



if __name__ == '__main__':
    n_inputs = 80
    n_validation = 100
    L = 6
    m = 40

    xlims = [0, 2*np.pi]
    ylims = [0, np.pi]

    a = lambda x, y: 0
    b = lambda x, y: 1 * torch.sin(3*y) * torch.exp(-y)
    c = lambda x, y: 1
    d = lambda x, y: torch.sin(3*x)
    e = lambda x, y: torch.sin(y / 2)
    bc_type = [1, 4, 1, 1]
    bcs = [d, d, a, a]

    f_torch = lambda u, x, y: -torch.sin(x)*torch.sin(y) - torch.cos(u)
    #f_torch = lambda u, x, y: torch.sin(x)*torch.cos(y) + torch.sin(u)
    #f_torch = lambda u, x, y: torch.sin(x)*torch.cos(y) + torch.exp(-u**2)*torch.cos(x)*torch.sin(y)
    #f_torch = lambda u, x, y: -u*((x-np.pi)**2 / (0.5)**2 + (y-np.pi/2)**2 / (0.25)**2 - 1 / (0.5)**2 - 1 / (0.25)**2) + 4*u
    f_np = lambda x, y: -np.sin(x)*np.sin(y)
    xx_val, yy_val, u_hat = train_func(n_inputs, n_validation, xlims, ylims,
                                       L, m, bc_type, bcs, f_torch, f_np, gamma=10,
                                       iterations=10000)
    y_exact = lambda x, y: 0.5*np.sin(x)*np.sin(y)
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(10, 6))
    #ax.plot_surface(xx_val, yy_val, y_exact(xx_val, yy_val))
    ax.plot_surface(xx_val, yy_val, u_hat.detach().numpy())
    #ax.set_title('Analytical solution', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel(r'$u$', fontsize=20)

    plt.show()
