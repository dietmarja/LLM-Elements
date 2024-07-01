# Stochastic Gradient Descent

The code in `stochastic_gradient_descent` implements a simple stochastic gradient descent algorithm to minimize the cost function $f(v) = v^2$.



\begin{table}[h!]
\centering
\begin{tabular}{|>{\raggedright}p{3.5cm}|>{\raggedright}p{5.5cm}|>{\raggedright}p{5.5cm}|}
\hline
\textbf{Aspect} & \textbf{Batch Gradient Descent} & \textbf{Stochastic Gradient Descent (SGD)} \\ \hline
Gradient Computation & Entire dataset & Single data point (or small batch) \\ \hline
Update Step & Once per epoch (after all data points) & After each data point (or small batch) \\ \hline
Stability and Accuracy & Smooth and stable updates & Noisy and erratic updates \\ \hline
Speed of Convergence & Slower for large datasets & Faster updates, potentially quicker convergence \\ \hline
Memory Usage & Higher (entire dataset in memory) & Lower (one data point or small batch) \\ \hline
Tendency to Escape Local Minima & Less likely & More likely due to noisier updates \\ \hline
\end{tabular}
\caption{Comparison of Batch Gradient Descent and Stochastic Gradient Descent}
\end{table}



## Gradient Function
The gradient of $f(v) = v^2$ is computed as $\nabla(v) = 2v$.

## Algorithm Steps

### 1. Initialization
- The parameter $\mathbf{v}$ is initialized with `initial_value`.
- A list `param_values` is created to store the parameter values at each iteration.

### 2. Iteration Loop
- **Loop Execution**: The loop runs for a maximum of `num_iterations` times.
- **Gradient Calculation**: The gradient $\nabla f(\mathbf{v})$ is computed using the provided `gradient_func`.
  - The parameter update step is calculated as:
    $\Delta \mathbf{v} = -\alpha \nabla f(\mathbf{v})$
  - Here, $\alpha$ is the learning rate.
- **Convergence Check**: The algorithm checks if the magnitude of the update step $\Delta \mathbf{v}$ is less than or equal to the tolerance `tol`.
  - If the condition is met, the loop breaks early, indicating convergence.
- **Parameter Update**: The parameter $\mathbf{v}$ is updated as: $\mathbf{v} \leftarrow \mathbf{v} + \Delta \mathbf{v}$
  - The updated parameter value is appended to `param_values`.

### 3. Return
- The function returns the optimized parameter value and the list of parameter values at each iteration.

## Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the gradient function separately
def compute_gradient(param):
    return 2 * param

# Stochastic Gradient Descent function
def stochastic_gradient_descent(gradient_func, initial_value, learning_rate, num_iterations=50, tol=1e-06):
    parameter = initial_value
    param_values = [parameter]
    for _ in range(num_iterations):
        # Simulate the selection of a random data point
        # Here we use the same gradient function for simplicity
        gradient = -learning_rate * gradient_func(parameter)
        if np.all(np.abs(gradient) <= tol):
            break
        parameter += gradient
        param_values.append(parameter)
    return parameter, param_values

# Run stochastic gradient descent with a start value of 17.0
optimized_value, param_trajectory = stochastic_gradient_descent(
    gradient_func=compute_gradient, initial_value=17.0, learning_rate=0.2
)

# Plot the points considered in finding the best value
plt.figure(figsize=(10, 6))
plt.plot(param_trajectory, 'bo-', label='Parameter values')
plt.axhline(y=optimized_value, color='r', linestyle='--', label=f'Optimized value: {optimized_value:.2f}')
plt.title('Stochastic Gradient Descent Progress')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
plt.show()

# Create a DataFrame to store the results
iterations = list(range(len(param_trajectory)))
results_df = pd.DataFrame({
    'Iteration': iterations,
    'Parameter Value': param_trajectory
})

# Display the DataFrame
print(results_df)
