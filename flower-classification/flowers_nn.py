from scipy.io import loadmat
import numpy as np
import random

# Define nonlinear activation
def sigmoid(z): return 1 / (1 + np.exp(-z))

# Construct neural network
def forward_propagation(X, W1, W2, N):
    # First layer
    Z1 = W1 @ X
    A = sigmoid(Z1)         # Nonlinear activation of hidden layer

    # Augment A = sigmoid(Z1) to account for biases
    A_aug = np.zeros((3, N))
    ones = np.ones((1, N))
    A_aug[0, :] = ones
    A_aug[1:, :] = A

    # Last layer
    Z2 = W2 @ A_aug
    yhat = sigmoid(Z2)      # Final activation
    return yhat, A, A_aug


# Define binary cross-entropy loss per sample
def total_loss(y, yhat, N):
    return -np.sum(y*np.log(yhat) + (1-yhat)*np.log(1-yhat)) / N

def back_propagation(x, y, yhat, W1, W2, A, A_aug, N):
    dZ2 = yhat - y
    dW2 = (dZ2 @ A_aug.T) / N           # Derivative of loss wrt W2

    dA_aug = W2.T @ (yhat - y)
    dA = dA_aug[1:, :]
    dW1 = ((dA * A * (1-A)) @ x.T) / N  # Derivative of loss wrt W1

    return dW1, dW2

def update_ws(W1, W2, dW1, dW2, rate):
    W1 -= rate * dW1
    W2 -= rate * dW2
    return W1, W2

def train(X, y, W1, W2, iterations, rate, N):
    for i in range(iterations):
        # Compute total loss
        yhat, A, A_aug = forward_propagation(X, W1, W2, N)
        loss = total_loss(y, yhat, N)

        # Backpropagation and updating weights & biases
        dW1, dW2 = back_propagation(X, y, yhat, W1, W2, A, A_aug, N)
        W1, W2 = update_ws(W1, W2, dW1, dW2, rate)

        if i % 10000 == 0:
            print(f'Iteration {i} - Loss: {loss:4f}')

    return W1, W2

