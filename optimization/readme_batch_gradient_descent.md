<h1 align="center">Batch Gradient Descent</h1>

The code in batch_gradient_descent implements a simple standard or batch gradient descent algorithm to minimize the cost function $f(v) = v^2$

* Gradient Function: The gradient of  $f(v) = v^2$  is computed as  $\text{gradient}(v) = 2v$.
* Batch Gradient Descent: The function (batch_gradient_descent) iteratively updates the parameter value to minimize the cost function.
  + It starts with an initial value (initial_value), a learning rate (learning_rate), and runs for a specified number of iterations (num_iterations).
  + The algorithm stops early if the change in parameter value is smaller than a specified tolerance (tol).
* Plotting: The progress of the parameter values over the iterations is plotted.
