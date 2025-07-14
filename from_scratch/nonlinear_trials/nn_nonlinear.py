import numpy as np


# Define nonlinear activation
def sigmoid(z): return 1 / (1 + np.exp(-z))


# Construct neural network
def forward_propagation(X, W1, W2, N, m):
    # Hidden layer
    Z1 = W1 @ X     # Linear transformation
    A = sigmoid(Z1) # Nonlinear activation

    # Augment A
    A_aug = np.zeros((m+1, N+1))
    A_aug[0, :] = np.ones((1, N+1))
    A_aug[1:, :] = A

    # Final layer
    y_hat = W2 @ A_aug  # Linear transformation
    return y_hat, A, A_aug


# Loss function
def compute_loss(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma):
    def second_der(X, W1, W2, N, m, stepsize):
        '''
        returns predictions and their first and second derivatives
        '''
        # Perturb inputs for the finite differences
        X_plus = np.copy(X)
        X_plus[1, :] += stepsize
        X_minus = np.copy(X)
        X_minus[1, :] -= stepsize

        # Obtain predictions for the perturbed & unperturbed inputs
        y_hat, _, _ = forward_propagation(X, W1, W2, N, m)
        y_plus, _, _ = forward_propagation(X_plus, W1, W2, N, m)
        y_minus, _, _ = forward_propagation(X_minus, W1, W2, N, m)

        # Return the derivative through finite differences
        return [y_hat, (y_hat - y_minus) / stepsize, (y_plus - 2 * y_hat + y_minus) / (stepsize ** 2)]

    # Call second_der() to get derivatives
    y_hat, y_1st, y_2nd = second_der(X, W1, W2, N, m, stepsize)
    f_vals = np.array([f(X[1, i], y_hat[0, i]) for i in range(1, X.shape[1]-1)])

    if typea == 1:
        loss_a = (y_hat[0, 0] - ya)**2
    elif typea == 2:
        loss_a = (y_1st[0, 0] - ya)**2
    else:
        loss_a = (y_hat[0, 0] + y_1st[0, 0] - ya)**2

    if typeb == 1:
        loss_b = (y_hat[0, -1] - yb)**2
    elif typeb == 2:
        loss_b = (y_1st[0, -1] - yb)**2
    else:
        loss_b = (y_hat[0, -1] + y_1st[0, -1] - yb)**2

    # Calculate the loss
    loss = np.sum((y_2nd[0, 1:N] - f_vals)**2) + gamma * (loss_a + loss_b)
    return loss, y_1st, y_2nd


def back_propagation(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma):
    ## Finite differences wrt x
    # Perturb inputs
    X_plus = np.copy(X)
    X_plus[1, :] += stepsize
    X_minus = np.copy(X)
    X_minus[1, :] -= stepsize

    # Run the unperturbed inputs through the net
    y_hat, A, A_aug = forward_propagation(X, W1, W2, N, m)
    # Compute the second derivatives
    _, y_1st, y_2nd = compute_loss(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)

    # Run the perturbed inputs through the net
    _, A_plus, A_augplus = forward_propagation(X_plus, W1, W2, N, m)
    _, A_minus, A_augminus = forward_propagation(X_minus, W1, W2, N, m)

    # Compute necessary arrays
    f_vals = np.array([f(X[1, i], y_hat[0, i]) for i in range(1, N)])
    w2_col = W2[0, 1:].T.reshape((m, 1))                    # Column vector storing values of the second weights
    residual = (y_2nd[0, 1:N] - f_vals).T.reshape((N-1, 1)) # Residuals of inner points

    ## Compute contribution to dL/dW1 from f(x, y)
    # Biases 1
    matrix_b1 = np.zeros((m, N-1)) + f_vals
    for j in range(m):  # For each bias
        # Perturb it
        W1_b1 = W1.copy()
        W1_b1[j, 0] += stepsize
        # Calculate y and f with the perturbation
        y_b1, _, _ = forward_propagation(X, W1_b1, W2, N, m)
        f_b1 = np.array([f(X[1, i], y_b1[0, i]) for i in range(1, N)])
        # Obtain the necessary array
        matrix_b1[j, :] -= f_b1
    # df/db1
    f_b1 = matrix_b1 @ residual
    # Weights 1
    matrix_w1 = np.zeros((m, N-1)) + f_vals
    for j in range(m):  # For each weight
        # Perturb it
        W1_w1 = W1.copy()
        W1_w1[j, 1] += stepsize
        # Calculate y and f with the perturbation
        y_w1, _, _ = forward_propagation(X, W1_w1, W2, N, m)
        f_w1 = np.array([f(X[1, i], y_w1[0, i]) for i in range(1, N)])
        # Obtain the necessary array
        matrix_w1[j, :] -= f_w1
    # df/dw1
    f_w1 = matrix_w1 @ residual
    # Then df/dW1
    dfdW1 = np.hstack((f_b1, f_w1))

    ## Obtain dL/dW1
    # Inner terms
    dA_minus = (A_minus[:, 1:N] * (1 - A_minus[:, 1:N])) @ (residual * X_minus[:, 1:N].T)
    dA = (A[:, 1:N] * (1 - A[:, 1:N])) @ (residual * X[:, 1:N].T)
    dA_plus = (A_plus[:, 1:N] * (1 - A_plus[:, 1:N])) @ (residual * X_plus[:, 1:N].T)
    dA_total = w2_col * (dA_plus - 2 * dA + dA_minus)
    dW1 = (2 / stepsize**2) * dA_total + (2 / stepsize) * dfdW1

    # Boundary terms
    if typea == 1:
        bt_w1_a = 2 * gamma * w2_col * ((y_hat[0, 0] - ya) * A[:, 0] * (1 - A[:, 0]) * X[1, 0]).reshape((m, 1))
        bt_b1_a = 2 * gamma * w2_col * ((y_hat[0, 0] - ya) * A[:, 0] * (1 - A[:, 0])).reshape((m, 1))
    elif typea == 2:
        bt_w1_a = ((2 * gamma / stepsize) * w2_col *
                   (y_1st[0, 0] - ya) * (A[:, 0] * (1 - A[:, 0]) * X[1, 0] -
                                         A_minus[:, 0] * (1 - A_minus[:, 0]) * (X[1, 0] - stepsize)))
        bt_b1_a = ((2 * gamma / stepsize) * w2_col *
                   (y_1st[0, 0] - ya) * (A[:, 0] * (1 - A[:, 0]) -
                                         A_minus[:, 0] * (1 - A_minus[:, 0])))

    if typeb == 1:
        bt_w1_b = 2 * gamma * w2_col * ((y_hat[0, -1] - yb) * A[:, -1] * (1 - A[:, -1]) * X[1, -1]).reshape((m, 1))
        bt_b1_b = 2 * gamma * w2_col * ((y_hat[0, -1] - yb) * A[:, -1] * (1 - A[:, -1])).reshape((m, 1))
    elif typeb == 2:
        bt_w1_b = ((2 * gamma / stepsize) * w2_col *
                   (y_1st[0, -1] - yb) * (A[:, -1] * (1 - A[:, -1]) * X[1, -1] -
                                          A_minus[:, -1] * (1 - A_minus[:, -1]) * (X[1, -1] - stepsize)))
        bt_b1_b = ((2 * gamma / stepsize) * w2_col *
                   (y_1st[0, -1] - yb) * (A[:, -1] * (1 - A[:, -1]) -
                                          A_minus[:, -1] * (1 - A_minus[:, -1])))


    bt_w1 = bt_w1_a + bt_w1_b
    bt_b1 = bt_b1_a + bt_b1_b
    # Total derivatives dL/dW1
    dW1[:, 0] += bt_b1[:, 0]
    dW1[:, 1] += bt_w1[:, 0]

    ## Compute contribution to dL/dW2 from f(x, y)
    # Bias 2
    matrix_b2 = f_vals.copy()
    # Perturb b2
    W2_b2 = W2.copy()
    W2_b2[0, 0] += stepsize
    # Calculate y and f with the perturbation
    y_b2, _, _ = forward_propagation(X, W1, W2_b2, N, m)
    f_b2 = np.array([f(X[1, i], y_b2[0, i]) for i in range(1, N)])
    matrix_b2 -= f_b2
    # df/db2
    f_b2 = matrix_b2 @ residual
    # Weights 2
    matrix_w2 = np.zeros((m, N - 1)) + f_vals
    for j in range(m):  # For each weight
        # Perturb it
        W2_w2 = W2.copy()
        W2_w2[0, j + 1] += stepsize
        # Calculate y and f with the perturbation
        y_w2, _, _ = forward_propagation(X, W1, W2_w2, N, m)
        f_w2 = np.array([f(X[1, i], y_w2[0, i]) for i in range(1, N)])
        # Obtain the necessary array
        matrix_w2[j, :] -= f_w2
    # df/dw2
    f_w2 = (matrix_w2 @ residual).T
    # Then df/dW2
    dfdW2 = np.hstack((f_b2, f_w2[0]))

    ## Obtain dL/dW2
    # Inner terms
    dws = (2 / stepsize**2) * ((A_plus[:, 1:N] - 2 * A[:, 1:N] + A_minus[:, 1:N]) @ residual).T
    dW2 = np.insert(dws, 0, 0).reshape((1, m+1))
    dW2 += (2 / stepsize) * dfdW2
    # Boundary terms
    if typea == 1:
        bt_w2_a = 2 * gamma * (y_hat[0, 0] - ya) * A[:, 0]
        bt_b2_a = 2 * gamma * (y_hat[0, 0] - ya)
    elif typea == 2:
        bt_w2_a = (2 * gamma / stepsize) * (y_1st[0, 0] - ya) * (A[:, 0] - A_minus[:, 0])
        bt_b2_a = 0

    if typeb == 1:
        bt_w2_b = 2 * gamma * (y_hat[0, -1] - yb) * A[:, -1]
        bt_b2_b = 2 * gamma * (y_hat[0, -1] - yb)
    elif typeb == 2:
        bt_w2_b = (2 * gamma / stepsize) * (y_1st[0, -1] - yb) * (A[:, -1] - A_minus[:, -1])
        bt_b2_b = 0

    bt_w2 = bt_w2_a + bt_w2_b
    bt_b2 = bt_b2_a + bt_b2_b
    # Total derivatives dL/dW2
    dW2[0, 1:] += bt_w2
    dW2[0, 0] += bt_b2

    return dW1, dW2


def train(X, f, W1, W2, tol, N, m, ya, typea, yb, typeb, stepsize, gamma):
    k = 1   # Iteration number
    # Compute loss gradients
    dW1, dW2 = back_propagation(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)
    # Early stopping count
    wait = 0
    back = 0
    # Early stopping patience
    patience = 1000

    # Initialise loss (temp. variable)
    lossnow = tol + 1

    # Iterate until all components of the gradient are <= tol, or loss is <= tol
    while ((np.linalg.norm(dW1) > tol) | (np.linalg.norm(dW2) > tol)) & (lossnow > tol):
        # Obtain the gradients of loss function
        dW1, dW2 = back_propagation(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)
        grad = np.concatenate((dW1.flatten(), dW2.flatten()))
        norm = np.linalg.norm(grad)

        # Define descent direction
        s1 = -dW1
        s2 = -dW2

        # Obtain an appropriate stepsize (backtracking Armijo)
        tau = np.random.rand() * 3 / 5  # number in (0,1) scaling down stepsize
        beta = np.random.rand()         # bArmijo parameter
        a = 1   # Stepsize along negative gradient direction

        # Compute current loss and loss assuming stepsize a
        lossnow, _, _ = compute_loss(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)
        lossnext, _, _ = compute_loss(X, W1-a*dW1, W2-a*dW2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)
        while lossnext > (lossnow - beta*a*(norm**2)):
            # Decrease a until sufficient decrease of loss is achieved
            a *= tau
            # Compute new losses (current and for new a)
            lossnext, _, _ = compute_loss(X, W1-a*dW1, W2-a*dW2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)
            lossnow, _, _ = compute_loss(X, W1, W2, N, m, stepsize, f, ya, typea, yb, typeb, gamma)

        # Obtain the parameter values for the next iteration
        W1 += a * s1
        W2 += a * s2

        # Print loss
        if (k % 50 == 0):
            print(f'Iteration {k}\t Loss: {lossnow:.9f}\n'
                  f'dW1={np.linalg.norm(dW1)}\ndW2={np.linalg.norm(dW2)}')

        # Early stopping mechanism
        if (lossnow - lossnext) < 1e-5:
            wait += 1
            back = 1
            if k % 20 == 0:
                print(f'Loss is not decreasing (for the {wait} time)')
            if wait >= patience:
                print(f'Optimisation STOPPED at iteration {k}')
                break
        else:
            back = 0

        if back == 0:
            wait = 0

        # Next iteration
        k += 1
    print(f'{k} iterations to reach a tolerance {tol}')

    return W1, W2, lossnow
