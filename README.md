<h1 align="center">Fine Tuning</h1>


Fine-tuning a pretrained large language model involves updating its weights 
so that the model can better handle the specific data it is being 
trained on. Key processing steps involve:

- Prepare the training loop
    - Initialize a pretrained model and a tokenizer via the transformer library from Huggingface
    - Standardize the length of input sequences via padding tokens
    - Make training data availablle
    - Define the optimizer (Adaptive Moment Estimation, Adam) and the loss function (Cross-Entropy)
- Run the training loop
    - An outer for loop runs over the epocs and an innner for loop iterates over batches

Next I describe a sequence of Jupyter Notebooks which carries out fine tuning. We start with a simple approach to fine-tuning and 
add more bells and whistles in more complex version of fine-tuning



### Fine_Tuning_01.ipynb
The training loop set up in Fine_Tuning_01.ipynb fine-tunes a pretrained model 
on a new (and very simple!) dataset. 

### Diagram for Fine_Tuning_01.ipynb

```mermaid
graph TD;
    A[Initialize Model and Tokenizer] --> B[Prepare Data];
    B --> C[Define Optimizer and Loss Function];
    C --> D[Start Training Loop];
    D --> E{More Epochs?};
    E -- Yes --> F[Train on Batch];
    F --> G[Compute Loss];
    G --> H[Backpropagate];
    H --> I[Update Parameters];
    I --> J{More Batches?};
    J -- Yes --> F;
    J -- No --> D;
```

### Training_Loop_02.ipynb
The training loop set up in Training_Loop_01.ipynb fine-tunes a pretrained model 
on a new (and very simple!) dataset. 

### Diagram for Training_Loop_02.ipynb

```mermaid
flowchart TD
    A[Start] --> B[Initialize Model and Tokenizer]
    B --> C{Pad Token Check}
    C -->|No Pad Token| D[Add Special Tokens]
    C -->|Pad Token Exists| E[Prepare Data]
    D --> E[Prepare Data]
    E --> F[Define Optimizer and Loss Function]
    F --> G[Training Loop]
    G --> H[Model Train Mode]
    H --> I[Loop Through Training Data]
    I --> J[Forward Pass]
    J --> K[Compute Loss]
    K --> L[Backpropagation]
    L --> M[Optimizer Step]
    M --> N{Check Batch}
    N -->|Print Loss Every 10 Batches| O[Print Loss]
    N --> P[End Batch]
    P --> Q{End of Epoch?}
    Q -->|Yes| R[Validation Loop]
    Q -->|No| I
    R --> S[Model Eval Mode]
    S --> T[Loop Through Validation Data]
    T --> U[Forward Pass on Validation Data]
    U --> V[Compute Validation Loss]
    V --> W[Calculate Average Validation Loss]
    W --> X{End of Training?}
    X -->|Yes| Y[Generate Sample Output]
    X -->|No| G
    Y --> Z[Print Sample Output]
    Z --> AA[Training Completed]
    P --> Q
```





