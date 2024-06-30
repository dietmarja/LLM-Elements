# Batch Gradient Descent

The code in `batch_gradient_descent` implements a simple standard or batch gradient descent algorithm to minimize the cost function $f(v) = v^2$.

## Gradient Function
The gradient of  $f(v) = v^2$ is computed as $\nabla(v) = 2v$.

## Algorithm Steps

### 1. Initialization
- The parameter \(\mathbf{v}\) is initialized with `initial_value`.
- A list `param_values` is created to store the parameter values at each iteration.

### 2. Iteration Loop
- **Loop Execution**: The loop runs for a maximum of `num_iterations` times.
- **Gradient Calculation**: The gradient $\nabla f(\mathbf{v})$ is computed using the provided `gradient_func`.
  - The parameter update step is calculated as:
    $\Delta \mathbf{v} = -\alpha \nabla f(\mathbf{v})$
  - Here, $\alpha$ is the learning rate.
- **Convergence Check**: The algorithm checks if the magnitude of the update step $\Delta \mathbf{v}$ is less than or equal to the tolerance `tol`.
  - If the condition is met, the loop breaks early, indicating convergence.
- **Parameter Update**: The parameter $\mathbf{v}$ is updated as:
    $ \mathbf{v} \leftarrow \mathbf{v} + \Delta \mathbf{v} $
  - The updated parameter value is appended to `param_values`.

### 3. Return
- The function returns the optimized parameter value and the list of parameter values at each iteration.

## Gradient Function Example
