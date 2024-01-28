import flowers_classification as fc
import flowers_nn as fn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import data
inputs = fc.inputs
labels = fc.labels

data_type0 = inputs[:, labels[0, :] == 0]
data_type1 = inputs[:, labels[0, :] == 1]

# Scatter plot with separating surface----------------
# Scatter plot
fig1, ax1 = plt.subplots(1, 1)
c1 = ax1.scatter(data_type0[0, :], data_type0[1, :], color='red', label='Type 0')
c2 = ax1.scatter(data_type1[0, :], data_type1[1, :], color='blue', label='Type 1')
# h1, l1 = c1.legend_elements()
# h2, l2 = c2.legend_elements()

# Create a meshgrid to find separating surface
x_min, x_max = inputs[0, :].min() - 0.3, inputs[0, :].max() + 0.3
y_min, y_max = inputs[1, :].min() - 0.3, inputs[1, :].max() + 0.3
X, Y = np.meshgrid(np.arange(x_min, x_max, 0.001),
                   np.arange(y_min, y_max, 0.001))
N_inputs = X.shape[0] * X.shape[1]
points = np.zeros((3, N_inputs))
points[0, :] = np.ones((1, N_inputs))
points[1:, :] = np.c_[X.ravel(), Y.ravel()].T

predictions, _, _ = fn.forward_propagation(points, fc.W1, fc.W2, N_inputs)
verdict = np.array([round(i) for i in predictions[0]])
verdict = verdict.reshape(X.shape)

# Plot separating surface
c = ax1.contour(X, Y, verdict, alpha=0.8)
c_line = Line2D([], [], color='green', lw=2)
ax1.set_title('Iris classification and separating curve')
ax1.legend([c1, c2, c_line], ['Type 0', 'Type 1', 'Separating curve'])

plt.show()
