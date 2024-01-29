from scipy.io import loadmat
import flowers_nn as fn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import data
data = loadmat('flowers_data.mat')
inputs = data['data'][:, 2:].T
labels = data['type'].T
W1 = np.genfromtxt('weights_biases/W1.csv', delimiter=',')
W2 = np.genfromtxt('weights_biases/W2.csv', delimiter=',')

data_type0 = inputs[:, labels[0, :] == 0]
data_type1 = inputs[:, labels[0, :] == 1]

# Initialise figure and axis
fig1, ax1 = plt.subplots(1, 1)

# Create a meshgrid to find separating surface
x_min, x_max = inputs[0, :].min() - 0.3, inputs[0, :].max() + 0.3
y_min, y_max = inputs[1, :].min() - 0.3, inputs[1, :].max() + 0.3
X, Y = np.meshgrid(np.arange(x_min, x_max, 0.01),
                   np.arange(y_min, y_max, 0.01))
N_inputs = X.shape[0] * X.shape[1]      # Number of cells in the meshgrid

# Reshape the meshgrid to fit the accepted inputs of the NN (array where every row looks like 1, length, width)
points = np.zeros((3, N_inputs))
points[0, :] = np.ones((1, N_inputs))
points[1:, :] = np.c_[X.ravel(), Y.ravel()].T

# Run the trained network for every point in the mesh and round the predictions
predictions, A, A_aug = fn.forward_propagation(points, W1, W2, N_inputs)
preds = predictions.reshape(X.shape)
verdict = np.array([round(i) for i in predictions])
verdict = verdict.reshape(X.shape)

# Scatter plot
c1 = ax1.scatter(data_type0[0, :], data_type0[1, :], color='red', label='Type 0')
c2 = ax1.scatter(data_type1[0, :], data_type1[1, :], color='blue', label='Type 1')

# Plot separating surface
c = ax1.contour(X, Y, verdict, alpha=0.8)
c_line = Line2D([], [], color='green', lw=2)
ax1.set_title('Iris classification and separating curve')
ax1.legend([c1, c2, c_line], ['Type 0', 'Type 1', 'Separating curve'])

# Histograms of inputs and evolution of outputs after each layer when the meshgrids X, Y are used as inputs
plt.figure(figsize=(15, 7))

ax2 = plt.subplot2grid((2, 3), (0, 0))
ax2.hist(X.ravel())
ax2.set_title('Petal Width of whole meshgrid')

ax3 = plt.subplot2grid((2, 3), (1, 0))
ax3.hist(Y.ravel())
ax3.set_title('Petal Length of whole meshgrid')

ax4 = plt.subplot2grid((2, 3), (0, 1))
ax4.hist(A[0, :])
ax4.set_title('Neuron 1')

ax5 = plt.subplot2grid((2, 3), (1, 1))
ax5.hist(A[1, :])
ax5.set_title('Neuron 2')

ax6 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
ax6.hist(predictions)
ax6.set_title('Predictions of whole meshgrid')

plt.suptitle('Outputs of each layer for a square meshgrid of inputs')

plt.show()
