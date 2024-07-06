# PyTorch's automatic differentiation engine `autograd`
 `pytorch.autograd` is PyTorch's workhorse when it comes to neural network training. 
 More specifically, `pytorch.autograd` computes gradients of tensor operations, allowing for efficient backpropagation during the training process.
The chart below illustrates the code in `autograd.ipynb` where autograd is used create a small neural network that predicts the next word in a sequence, which is a fundamental task in language modeling:


```mermaid
graph TD
    A[Start] --> B[Initialize Model Parameters]
    B --> C[Create SimpleLanguageModel]
    C --> D[Define Loss Function CrossEntropyLoss]
    D --> E[Create Optimizers Adam and SGD]
    E --> F[Training Loop]
    F --> G[Generate Input Data]
    G --> H[Forward Pass]
    H --> I[Compute Loss]
    I --> J[Backward Pass Autograd]
    J --> K[Update Parameters]
    K --> L{Epoch Complete?}
    L -->|No| G
    L -->|Yes| M[Visualize Computational Graph]
    M --> N[Plot Loss over Time]
    N --> O[Visualize Word Embeddings]
    O --> P[End]

    subgraph SimpleLanguageModel
    Q[Embedding Layer]
    R[LSTM Layer]
    S[Fully Connected Layer]
    Q --> R --> S
    end

    subgraph Training Process
    G --> H --> I --> J --> K
    end

    subgraph Visualizations
    M
    N
    O
end
```


The code generates three charts:<br>

* model_graph.png: A visualization of the model's computational graph.
* loss_plot.png: A plot showing how the loss decreases over time for both Adam and SGD optimizers.
* word_embeddings.png: A 2D visualization of the learned word embeddings.


# References
A Gentle Introduction to torch.autograd<br>
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
