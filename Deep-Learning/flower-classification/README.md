This was my first task in a DL project I did in Oxford. In order to properly understand neural networks and backpropagation, I created this neural network from scratch, including only one hidden layer. SGD has not yet been implemented.

The file "flowers_data.mat" contains the input-ouput data.

The file "flowers_nn.py" contains only functions I designed for this specific NN.

The file "flowers_classification.py" uses the neural network. It takes tha data from "flowers_data.mat", splits it into a train set (80% of total data) and a test set (20%), trains the NN in "flowers_nn.py", and then tests it in the test set.

The file "flowers_graphs.py", based on the training of the NN from "flowers_classification.py", calculates the separating curve between both types of flowers. It then produces a scatter plot of the given data and the separating curve. Different separating curves produced by the randomness of the train/test generation can be accessed in "iris-classification-scatterplot.png", "iris-classification-curve.png", "iris-classification-curve2.png", "iris-classification-curve3.png", in the "graphs" directory.
However, in order to reduce the run time of this file, some already trained weights and biases ("W1.csv" and "W2.csv") are used instead of importing "flowers_classification.py" every time. As a consequence, the randomness in the separating curve is not present in this version of the code (hence the "classification_random_initialisation/" directory). Finally, this file also produces a sequence of histograms where a whole meshgrid is plotted after every layer in the NN, to capture the evolution of the binary classification. The sequence of histograms can be found in "layer_effect_meshgrid.png" in the directory "graphs".
