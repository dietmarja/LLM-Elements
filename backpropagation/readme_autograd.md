# PyTorch’s automatic differentiation engine `autograd`
 `pytoch.autograd` is the PyTorch’s  workhorse when it come to neural network training. 
 More specifically,  `pytoch.autograd`  computes gradients of tensor operations, allowing for efficient backpropagation during the training process.

Below is a simple example that demonstrates the use of autograd in a context typical forlanguage models. 
We'll create a small neural network that predicts the next word in a sequence, which is a fundamental task in language modeling:

To illustrate what the code does three files are generated

It generates three files:

+ model_graph.png: A visualization of the model's computational graph.
+ loss_plot.png: A plot showing how the loss decreases over time for both Adam and SGD optimizers.
+ word_embeddings.png: A 2D visualization of the learned word embeddings.


# References
A Gentle Introduction to torch.autograd<br>
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
