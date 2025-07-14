import numpy as np
import matplotlib.pyplot as plt


# Define nonlinear activation
def sigmoid(z): return 1 / (1 + np.exp(-z))


# Construct neural network
def forward_propagation(X, W1, W2, N, m):
    # First layer
    Z1 = W1 @ X
    A = sigmoid(Z1)

    A_aug = np.zeros((m+1, N+1))
    A_aug[0, :] = np.ones((1, N+1))
    A_aug[1:, :] = A

    y_hat = W2 @ A_aug
    return y_hat, A, A_aug


# Define loss function
## loss sums over differences of second derivative
def compute_loss(X, W1, W2, N, m, stepsize, f, ya, yb, gamma):

    def second_der(X, W1, W2, N, m, stepsize):
        '''
        returns predictions and second derivatives of predictions
        '''
        # Perturb inputs for the finite differences
        X_plus = np.copy(X)
        X_plus[1, :] += stepsize

        X_minus = np.copy(X)
        X_minus[1, :] -= stepsize

        # Obtain predictions for the perturbed inputs
        y_hat, _, _ = forward_propagation(X, W1, W2, N, m)
        y_plus, _, _ = forward_propagation(X_plus, W1, W2, N, m)
        y_minus, _, _ = forward_propagation(X_minus, W1, W2, N, m)

        # Calculate the second derivative through finite differences
        return [y_hat, (y_plus - 2 * y_hat + y_minus) / (stepsize ** 2)]

    y_hat, y_2nd = second_der(X, W1, W2, N, m, stepsize)
    f_vals = np.array([f(X[1, i], y_hat[0, i]) for i in range(1, X.shape[1]-1)])

    loss = np.sum((y_2nd[0, 1:N] - f_vals)**2) + gamma * ((y_hat[0, 0] - ya)**2 + (y_hat[0, -1] - yb)**2)
    return loss, y_2nd


def back_propagation(X, W1, W2, N, m, stepsize, f, ya, yb, gamma):
    # Initialise slightly perturbed inputs (for the finite differences)
    X_plus = np.copy(X)
    X_plus[1, :] += stepsize
    X_minus = np.copy(X)
    X_minus[1, :] -= stepsize

    # Run both perturbed, and the nonperturbed inputs through the net
    y_plus, A_plus, A_augplus = forward_propagation(X_plus, W1, W2, N, m)
    y_hat, A, A_aug = forward_propagation(X, W1, W2, N, m)
    y_minus, A_minus, A_augminus = forward_propagation(X_minus, W1, W2, N, m)

    # Compute the second derivatives
    _, y_2nd = compute_loss(X, W1, W2, N, m, stepsize, f, ya, yb, 1)

    # Initialise array with values of f(x, y) for each (inner) point in x
    f_vals = np.array([f(X[1, i], y_hat[0, i]) for i in range(1, X.shape[1]-1)])

    w2_col = W2[0, 1:].T.reshape((m, 1))                    # Column vector storing values of the second weights
    residual = (y_2nd[0, 1:N] - f_vals).T.reshape((N-1, 1)) # Residuals of inner points

    # Obtain dL/dW1
    # Inner points
    dA_minus = (A_minus[:, 1:N] * (1 - A_minus[:, 1:N])) @ (residual * X_minus[:, 1:N].T)
    dA = (A[:, 1:N] * (1 - A[:, 1:N])) @ (residual * X[:, 1:N].T)
    dA_plus = (A_plus[:, 1:N] * (1 - A_plus[:, 1:N])) @ (residual * X_plus[:, 1:N].T)
    dA_total = w2_col * (dA_plus - 2 * dA + dA_minus)
    dW1 = (2 / stepsize**2) * dA_total
    # Boundary terms
    bt_w1 = 2 * gamma * w2_col * ((y_hat[0, 0] - ya) * A[:, 0] * (1 - A[:, 0]) * X[1, 0] +
                                  (y_hat[0, -1] - yb) * A[:, N] * (1 - A[:, N]) * X[1, -1]).reshape((m, 1))
    bt_b1 =  2 * gamma * w2_col * ((y_hat[0, 0] - ya) * A[:, 0] * (1 - A[:, 0]) +
                                  (y_hat[0, -1] - yb) * A[:, N] * (1 - A[:, N])).reshape((m, 1))
    # Total derivatives dL/dW1
    dW1[:, 0] += bt_b1[:, 0]
    dW1[:, 1] += bt_w1[:, 0]

    # Obtain dL/dW2
    dws = (2 / stepsize**2) * ((A_plus[:, 1:N] - 2 * A[:, 1:N] + A_minus[:, 1:N]) @ residual).T
    dW2 = np.insert(dws, 0, 0).reshape((1, m+1))
    # Boundary terms
    bt_w2 = 2 * gamma * ((y_hat[0, 0] - ya) * A[:, 0] + (y_hat[0, -1] - yb) * A[:, N])
    bt_b2 = 2 * gamma * (y_hat[0, 0] - ya + y_hat[0, -1] - yb)
    # Total derivatives dL/dW2
    dW2[0, 1:] += bt_w2
    dW2[0, 0] += bt_b2

    return dW1, dW2


def update_ws(W1, W2, dW1, dW2, rate):
    W1 -= rate * dW1
    W2 -= rate * dW2
    return W1, W2


def train(X, f, W1, W2, iterations, rate, N, m, ya, yb, stepsize, gamma):
    for i in range(iterations):
        # Compute total loss
        loss, _ = compute_loss(X, W1, W2, N, m, stepsize, f, ya, yb, gamma)

        # Backpropagation and updating weights & biases
        dW1, dW2 = back_propagation(X, W1, W2, N, m, stepsize, f, ya, yb, gamma)
        W1, W2 = update_ws(W1, W2, dW1, dW2, rate)

        if (i % 500 == 0):
            print(f'Iteration {i} - Loss: {loss:4f}')

    return W1, W2
