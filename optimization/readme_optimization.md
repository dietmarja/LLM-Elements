# Optimization Algorithms

Optimization algorithms are key elements in machine learning. 
To a large degree the intelligence of an AI model is represented by the model parameters (weights and biased), and optimization algorithms allows us to idenfiy them. 
The  programs in the  `optimization` folder introduce 4 essential optimization algorithms. The table below offers a bird's eye overview of them

|         | **Batch Gradient Descent (BGD)** | **Stochastic Gradient Descent (SGD)** | **Mini-Batch Gradient Descent** | **AdamW** |
|----------------------------------|-----------------------------------------|----------------------------------------|------------------------------------|--------|
| **Gradient Computation**   | Entire dataset       | Single data point | Small batch of data points | Uses moving averages of gradients |
| **Update Step**     | Once per epoch (after all data points) | After each data point (or small batch) | After each mini-batch | Every iteration (uses accumulated gradients) |
| **Stability and Accuracy**  | Smooth and stable updates    | Noisy and erratic updates    | Balance between SGD and BGD | Generally stable with adaptive learning rates |
| **Memory Usage**     | Higher (entire dataset in memory)  | Lower (one data point or small batch) | Lower than BGD | Moderate (uses memory for accumulated gradients) |
| **Tendency to Escape Local Minima** | Less likely        | More likely due to noisier updates  | Less likely than SGD | Can aid in escaping local minima |
| **Speed** | Slow for large datasets | Fast updates, can be faster convergence | Faster than BGD, slower than SGD | Can be faster than SGD for noisy gradients |
| **Accuracy** | Can achieve high accuracy | May struggle with local minima | Often achieves good accuracy | Often achieves good accuracy |


```mermaid
graph LR
A[Start] --> B{Calculate Gradient}
B --> C{Update Weights}
C --> D{Evaluate Loss}
D --> E[Yes (Loss Reduced)]
D --> A[No (Increase Learning Rate)]
E --> F[Stop]


````
