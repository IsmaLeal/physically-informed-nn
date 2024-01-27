This was my first task in a DL project I did in Oxford. In order to properly understand neural networks and backpropagation, I created this neural network from scratch, including only one hidden layer. SGD has not yet been implemented.

The file "flowers_data.mat" contains the input-ouput data.

The file "flowers_nn.py" contains only functions I designed for this specific NN.

The file "flowers_classification.py" uses the neural network. It takes tha data from "flowers_data.mat", splits it into a train and a test set, trains the NN built in "flowers_nn.py", and then tests it in the test set.
