## Comparison of Gradient Descent Variants

| Aspect              | Batch Gradient Descent (BGD)         | Stochastic Gradient Descent (SGD)    | Mini-Batch Gradient Descent | AdamW |
|----------------------------------|-----------------------------------------|----------------------------------------|------------------------------------|--------|
| **Gradient Computation**     | Entire dataset             | Single data point  | Small batch of data points | Uses moving averages of gradients |
| **Update Step**         | Once per epoch (after all data points) | After each data point (or small batch) | After each mini-batch | Every iteration (uses accumulated gradients) |
| **Stability and Accuracy**    | Smooth and stable updates        | Noisy and erratic updates       | Balance between SGD and BGD | Generally stable with adaptive learning rates |
| **Speed of Convergence**     | Slower for large datasets        | Faster updates, potentially quicker convergence | Faster than BGD, slower than SGD | Can be faster than SGD for noisy gradients |
| **Memory Usage**         | Higher (entire dataset in memory)    | Lower (one data point or small batch) | Lower than BGD | Moderate (uses memory for accumulated gradients) |
| **Tendency to Escape Local Minima** | Less likely               | More likely due to noisier updates   | Less likely than SGD | Can aid in escaping local minima |