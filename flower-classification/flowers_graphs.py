from scipy.io import loadmat
import flowers_nn as fn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import data
data = loadmat('flowers_data.mat')
inputs = data['data'][:, 2:].T
labels = data['type'].T
W1 = np.genfromtxt('W1.csv', delimiter=',')
W2 = np.genfromtxt('W2.csv', delimiter=',')

data_type0 = inputs[:, labels[0, :] == 0]
data_type1 = inputs[:, labels[0, :] == 1]

# Initialise figure and axis
fig1, ax1 = plt.subplots(1, 1)

# Create a meshgrid to find separating surface
x_min, x_max = inputs[0, :].min() - 0.3, inputs[0, :].max() + 0.3
y_min, y_max = inputs[1, :].min() - 0.3, inputs[1, :].max() + 0.3
X, Y = np.meshgrid(np.arange(x_min, x_max, 0.001),
                   np.arange(y_min, y_max, 0.001))
N_inputs = X.shape[0] * X.shape[1]      # Number of cells in the meshgrid

# Reshape the meshgrid to fit the accepted inputs of the NN (array where every row looks like 1, length, width)
points = np.zeros((3, N_inputs))
points[0, :] = np.ones((1, N_inputs))
points[1:, :] = np.c_[X.ravel(), Y.ravel()].T

# Run the trained network for every point in the mesh and round the predictions
predictions, _, _ = fn.forward_propagation(points, W1, W2, N_inputs)
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

plt.show()
