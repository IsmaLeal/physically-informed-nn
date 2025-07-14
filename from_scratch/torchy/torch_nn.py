import numpy as np
from scipy.integrate import solve_bvp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stop_training = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if (self.wait >= self.patience):
                self.stop_training = True
# nn.Module: fundamental base class for all NN
class OdeNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, neurons, output_size):
        # Ensure PyTorch initialises all parts of the updated feedforward net
        super(OdeNN, self).__init__()

        # Create list 'layers' to hold the layers
        self.layers = nn.ModuleList()

        # Append input layer
        self.layers.append(nn.Linear(input_size, neurons))

        # Append hidden layers
        for i in range(hidden_size-1):
            self.layers.append(nn.Linear(neurons, neurons))

        # Append output layer
        self.layers.append(nn.Linear(neurons, output_size))

        # Custom weights initialisation for tanh and sigmoid
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        for layer in self.layers[:-1]:
            # Un-comment only one activation function
            #x = torch.sigmoid(layer(x))
            x = torch.tanh(layer(x))

        # No activation for output layer
        x = self.layers[-1](x)
        return x


class PdeNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, neurons_per_layer, output_size):
        super(PdeNN, self).__init__()
        # Create list to hold the layers
        self.layers = nn.ModuleList()
        # Append input layer
        self.layers.append(nn.Linear(input_size, neurons_per_layer))
        # Append hidden layers (all with same number of neurons)
        for i in range(hidden_size - 1):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        # Append output layer
        self.layers.append(nn.Linear(neurons_per_layer, output_size))
        # Custom weights initialisation
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x, y):
        '''
        :param x: 2-d torch.meshgrid of x points
        :param y: 2-d torch.meshgrid of y points
        :return: 2-d torch tensor of predictions
        '''
        # Create a vector of all inputs
        xy = torch.stack((x.flatten(), y.flatten()), dim=1)
        # Nonlinear activation
        for layer in self.layers[:-1]:
            # Un-comment only one activation function
            # xy = torch.sigmoid(layer(x))
            xy = torch.tanh(layer(xy))
        # No activation for output layer
        xy = self.layers[-1](xy)

        # Return as array of original shape ??????
        xy = xy.view(x.shape[0], x.shape[1], x.shape[2])
        return xy


def ODE_loss(y_hat, x, f, bc_type, y_a, y_b, gamma):
    # Ensure y_hat and x are tensors and enable gradient computation for x
    #x.requires_grad_(True)

    # Compute the first derivative of y_hat with respect to x
    y_1st = torch.autograd.grad(y_hat, x, grad_outputs=torch.ones_like(y_hat),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]

    # Compute the second derivative of y_hat with respect to x
    y_2nd = torch.autograd.grad(y_1st, x, grad_outputs=torch.ones_like(y_1st),
                                create_graph=True, retain_graph=True, allow_unused=True)[0]

    # Compute the inner loss as the mean squared error between y_2nd and f(x)
    inner_loss = torch.sum((y_2nd - f(x, y_hat, y_1st)) ** 2)

    # Compute the left boundary term loss
    if bc_type[0] == 1:
        bt_a = gamma * (y_hat[0] - y_a) ** 2
    elif bc_type[0] == 2:
        bt_a = gamma * (y_1st[0] - y_a) ** 2
    elif bc_type[0] == 3:
        bt_a = gamma * (y_hat[0] + y_1st[0] - y_a) ** 2

    # Compute the right boundary term loss
    if  bc_type[1] == 1:
        bt_b = gamma * (y_hat[-1] - y_b) ** 2
    elif bc_type[1] == 2:
        bt_b = gamma * (y_1st[-1] - y_b) ** 2
    elif bc_type[1] == 3:
        bt_b = gamma * (y_hat[-1] + y_1st[-1] - y_b) ** 2

    # Total boundary loss
    bt_loss = bt_a + bt_b

    # Total loss
    total_loss = inner_loss + bt_loss
    return total_loss


def PDE_loss(u_hat, xx, yy, f, bc_types, bcs, gamma):
    '''
    :param u_hat:
    :param xx:
    :param yy:
    :param f:
    :param bc_types: array of 4 boundary types (1: Dirichlet, 2: Von Neumann wrt x,
    3: Von Neumann wrt y, 4: Robin wrt x, 5: Robin wrt y) in the order: bottom, top, left, right
    :param bcs: array of 4 anonymous functions in order: bottom, top, left, right
    :param gamma:
    :return:
    '''
    # Derivatives
    u = u_hat
    u_x = torch.autograd.grad(u, xx, grad_outputs=torch.ones_like(u_hat),
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x, xx, grad_outputs=torch.ones_like(u_hat),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]

    u_y = torch.autograd.grad(u, yy, grad_outputs=torch.ones_like(u_hat),
                              create_graph=True, retain_graph=True, allow_unused=True)[0]
    u_yy = torch.autograd.grad(u_y, yy, grad_outputs=torch.ones_like(u_hat),
                               create_graph=True, retain_graph=True, allow_unused=True)[0]

    ## BOUNDARY LOSS
    bt_loss = 0
    #Bottom boundary
    if bc_types[0] == 1:
        bt_loss += ((u_hat[0, :] - bcs[0](xx[0, :], yy[0, :])) ** 2).sum()
    elif bc_types[0] == 2:
        bt_loss += ((u_x[0, :] - bcs[0](xx[0, :], yy[0, :])) ** 2).sum()
    elif bc_types[0] == 3:
        bt_loss += ((u_y[0, :] - bcs[0](xx[0, :], yy[0, :])) ** 2).sum()
    elif bc_types[0] == 4:
        bt_loss += ((u_hat[0, :] + u_x[0, :] - bcs[0](xx[0, :], yy[0, :])) ** 2).sum()
    elif bc_types[0] == 5:
        bt_loss += ((u_hat[0, :] + u_y[0, :] - bcs[0](xx[0, :], yy[0, :])) ** 2).sum()
    # Top boundary
    if bc_types[1] == 1:
        bt_loss += ((u_hat[-1, :] - bcs[1](xx[-1, :], yy[-1, :])) ** 2).sum()
    elif bc_types[1] == 2:
        bt_loss += ((u_x[-1, :] - bcs[1](xx[-1, :], yy[-1, :])) ** 2).sum()
    elif bc_types[1] == 3:
        bt_loss += ((u_y[-1, :] - bcs[1](xx[-1, :], yy[-1, :])) ** 2).sum()
    elif bc_types[1] == 4:
        bt_loss += ((u_hat[-1, :] + u_x[-1, :] - bcs[1](xx[-1, :], yy[-1, :])) ** 2).sum()
    elif bc_types[1] == 5:
        bt_loss += ((u_hat[-1, :] + u_y[-1, :] - bcs[1](xx[-1, :], yy[-1, :])) ** 2).sum()
    # Left boundary
    if bc_types[2] == 1:
        bt_loss += ((u_hat[:, 0] - bcs[2](xx[:, 0], yy[:, 0])) ** 2).sum()
    elif bc_types[2] == 2:
        bt_loss += ((u_x[:, 0] - bcs[2](xx[:, 0], yy[:, 0])) ** 2).sum()
    elif bc_types[2] == 3:
        bt_loss += ((u_y[:, 0] - bcs[2](xx[:, 0], yy[:, 0])) ** 2).sum()
    elif bc_types[2] == 4:
        bt_loss += ((u_hat[:, 0] + u_x[:, 0] - bcs[2](xx[:, 0], yy[:, 0])) ** 2).sum()
    elif bc_types[2] == 5:
        bt_loss += ((u_hat[:, 0] + u_y[:, 0] - bcs[2](xx[:, 0], yy[:, 0])) ** 2).sum()
    # Right boundary
    if bc_types[3] == 1:
        bt_loss += ((u_hat[:, -1] - bcs[3](xx[:, -1], yy[:, -1])) ** 2).sum()
    elif bc_types[3] == 2:
        bt_loss += ((u_x[:, -1] - bcs[3](xx[:, -1], yy[:, -1])) ** 2).sum()
    elif bc_types[3] == 3:
        bt_loss += ((u_y[:, -1] - bcs[3](xx[:, -1], yy[:, -1])) ** 2).sum()
    elif bc_types[3] == 4:
        bt_loss += ((u_hat[:, -1] + u_x[:, -1] - bcs[3](xx[:, -1], yy[:, -1])) ** 2).sum()
    elif bc_types[3] == 5:
        bt_loss += ((u_hat[:, -1] + u_y[:, -1] - bcs[3](xx[:, -1], yy[:, -1])) ** 2).sum()

    bt_loss *= gamma

    # INNER LOSS
    inner_loss = ((u_xx[1:-1, 1:-1] + u_yy[1:-1, 1:-1] - f(u_hat[1:-1, 1:-1], xx[1:-1, 1:-1], yy[1:-1, 1:-1])) ** 2).sum()

    # TOTAL LOSS
    total_loss = bt_loss + inner_loss
    return total_loss



def ODE_training(net, x, x_val, loss, optimiser, iterations, f, bc, bc_type, gamma,
             validate_every=50, loss_vs_iterations=False):
    x = x.detach().requires_grad_(True)
    x_val = x_val.detach().requires_grad_(True)
    early_stopping = EarlyStopping(patience=150, min_delta=0.0001)

    val_losses = []
    epochs = []
    for iteration in range(iterations):
        # ensure no residual gradient information from previous epochs and the outputs can be differentiated wrt x
        optimiser.zero_grad()
        net.train()

        # Forward pass
        y_hat = net(x)

        # Compute loss
        total_loss = loss(y_hat, x, f, bc_type, bc[0], bc[1], gamma)

        # Compute gradient of loss wrt all parameters with requires_grad=True
        total_loss.backward()
        optimiser.step()

        if iteration % validate_every == 0:
            net.eval()

            y_hat_val = net(x_val)
            val_loss = loss(y_hat_val, x_val, f, bc_type, bc[0], bc[1], gamma)

            val_losses.append(val_loss.item())
            epochs.append(iteration)

            # Early stopping check
            early_stopping(val_loss.item())
            if early_stopping.stop_training:
                print(f'Stopping at iteration {iteration+1}')
                break
            net.train()

        if iteration % 500 == 0:
            print(f'Iteration {iteration+1}, loss: {total_loss.item()}\nValidation loss: {val_loss.item()}')
    x.requires_grad = False
    x_val.requires_grad = False
    print(f'total loss: {total_loss.item()}; val loss: {val_loss.item()}')
    if not loss_vs_iterations:
        return total_loss.item(), val_loss.item()
    else:
        return total_loss.item(), val_loss.item(), val_losses, epochs


def PDE_training(net, xx, yy, xx_val, yy_val, loss, optimiser, iterations, f, bcs, bc_type, gamma,
             validate_every=50, loss_vs_iterations=False):
    # Allow gradients wrt xx and yy for loss
    xx = xx.detach().requires_grad_(True)
    xx_val = xx_val.detach().requires_grad_(True)
    yy = yy.detach().requires_grad_(True)
    yy_val = yy_val.detach().requires_grad_(True)
    early_stopping = EarlyStopping(patience=150, min_delta=0.0001)

    val_losses = []
    epochs = []
    for iteration in range(iterations):
        # ensure no residual gradient information from previous epochs and the outputs can be differentiated wrt x
        optimiser.zero_grad()
        net.train()

        # Forward pass
        u_hat = net(xx, yy)

        # Compute loss
        total_loss = loss(u_hat, xx, yy, f, bc_type, bcs, gamma)

        # Compute gradient of loss wrt all parameters with requires_grad=True
        total_loss.backward()
        optimiser.step()
        #x.detach()

        if iteration % validate_every == 0:
            net.eval()
            #xx_val.requires_grad_(True)
            #yy_val.requires_grad_(True)

            u_hat_val = net(xx_val, yy_val)
            val_loss = loss(u_hat_val, xx_val, yy_val, f, bc_type, bcs, gamma)

            val_losses.append(val_loss.item())
            epochs.append(iteration)

            # Early stopping check
            early_stopping(val_loss.item())
            if early_stopping.stop_training:
                print(f'Stopping at iteration {iteration+1}')
                break
            net.train()

        if iteration % 10 == 0:
            print(f'Iteration {iteration+1}, loss: {total_loss.item()}\nValidation loss: {val_loss.item()}')
    xx.requires_grad = False
    yy.requires_grad = False
    xx_val.requires_grad = False
    yy_val.requires_grad = False
    print(f'total loss: {total_loss.item()}; val loss: {val_loss.item()}')
    if not loss_vs_iterations:
        return total_loss.item(), val_loss.item()
    else:
        return total_loss.item(), val_loss.item(), val_losses, epochs


if __name__ == '__main__':
    # Boundary conditions
    x_min = float(input('Leftmost value of x: '))
    x_max = float(input('Rightmost value of x: '))

    bc_names = {
        1: 'y',
        2: 'y\'',
        3: 'y + y\''
    }

    bctype_left = int(input('Left BC type:\n1. Dirichlet\n2. Von Neumann\n3. Robin\n'))
    bctype_right = int(input('Right BC type:\n1. Dirichlet\n2. Von Neumann\n3. Robin\n'))
    if (bctype_left not in bc_names.keys()) | (bctype_right not in bc_names.keys()):
        raise ValueError('Your input was not one of the valid options.')
    BC_type = [bctype_left, bctype_right]

    bc_left = float(input(f'{bc_names[bctype_left]}({x_min}) = '))
    bc_right = float(input(f'{bc_names[bctype_right]}({x_max}) = '))
    BCs = [bc_left, bc_right]

    # Inputs
    x_vals = torch.linspace(x_min, x_max, 50).unsqueeze(1)
    x_np = x_vals.detach().numpy().flatten()

    # RHS of y'' = f(x)
    f_torch = lambda x, y, y1st: 3*y1st - y + torch.cos(x)
    f_np = lambda x, y, y1st: 3*y1st - y + np.cos(x)


    hidden_layers = 5
    m = 10
    # Creating instance of the network
    net = OdeNN(1, hidden_layers, m,1)
    optimiser = torch.optim.Adam(net.parameters(), 5e-3)
    iterations = 80
    gamma = 10

    # Training
    ODE_training(net, x_vals, ODE_loss, optimiser=optimiser,
                     iterations=iterations, f=f_torch, bc=BCs, bc_type=BC_type,
                     gamma=gamma)

    # Apply the model
    with torch.no_grad():
        predictions = net(x_vals)
    x_vals.requires_grad = True
    x_vals.detach()
    loss_now = ODE_loss(predictions, x_vals, f_torch, BC_type, BCs[0], BCs[1], gamma)

    # Analytical solution
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

    y_exact = np.zeros((2, x_np.size))
    sol_exact = solve_bvp(sys, bc, x_np, y_exact)

    plt.plot(x_np, sol_exact.sol(x_np)[0], label='Analytical solution', color='blue')
    plt.scatter(x_np, predictions.detach().numpy(), label='Predictions', color='orange')
    plt.legend()
    plt.show()
