import flowers_nn
from scipy.io import loadmat
import numpy as np
import random

# Import data
data = loadmat('flowers_data.mat')
inputs = data['data'][:, 2:].T
labels = data['type'].T
N = labels.shape[1] # Number of data sets

# Create a train set (80% of original) and augment the input
N_train = 80
train_indices = sorted(random.sample(range(100), N_train))
labels_train = [labels[0, i] for i in train_indices]    # 1xN_train array
X_train = np.zeros((3, N_train))
ones_train = np.ones((1, N_train))
X_train[0, :] = ones_train
X_train[1:, :] = inputs[:, train_indices]               # 3xN_train array

# Create a test set and augment the input
N_test = N - N_train
test_indices = [i for i in range(N) if i not in train_indices]
labels_test = [labels[0, i] for i in test_indices]      # 1xN_test array
X_test = np.zeros((3, N_test))
ones_test = np.ones((1, N_test))
X_test[0, :] = ones_test
X_test[1:, :] = inputs[:, test_indices]                 # 3xN_test array

# Randomly initialise weights & biases
W1 = np.random.rand(2, 3)   # Weights & biases for hidden layer
W2 = np.random.rand(1, 3)   # Weights for final layer

if __name__ == '__main__':
    isMain = True
else:
    isMain = False

# Train the network
yhat, A, A_aug = flowers_nn.forward_propagation(X_train, W1, W2, N_train)           # Obtain initial predictions
W1, W2 = flowers_nn.train(X_train, labels_train, W1, W2,
                          80000, 0.1, N_train, isMain=isMain)         # Train parameters

# Test the network
predictions, _, _ = flowers_nn.forward_propagation(X_test, W1, W2, N_test)          # Test predictions
verdict = [round(i) for i in predictions[0]]                                        # Round predictions
performance = [verdict[i] == labels_test[i] for i in range(len(labels_test))]       # Boolean list with successes

if __name__ == '__main__':
    print(f'Predictions: {predictions}\nActual results: {labels_test}')
    print(f'Success predicting on the test set: {performance}')
    print(f'Accuracy: {np.mean(performance)}')

    # Save the trained weights and biases as .csv files
    np.savetxt('W1.csv', W1, delimiter=',')
    np.savetxt('W2.csv', W2, delimiter=',')

