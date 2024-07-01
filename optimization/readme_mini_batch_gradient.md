# Mini-Batch Gradient

Mini-batch gradient descent is the most frequently used variant of gradient descent for training deep learning models. 
It offers a good compromise between efficiency, stability, parallelization, and memory usage, which is why it is the preferred 
choice for training large-scale deep learning models.

The code in `mini_batch_gradient_descent.ipynb` implements a simple neural network with one hidden layer and uses mini-batch gradient descent to train it. 
The objective is to minimize the Mean Squared Error (MSE) loss between the network’s predictions and the true output values. 
The network is trained on randomly generated data, and the training progress is visualized by plotting the loss over iterations.

