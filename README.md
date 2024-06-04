# Fine Tuning

Fine-tuning involves updating the weights of a pretrained model so that it can better handle the specific data it is being 
trained on. Essential processing steps involve:

- Prepare the training loop
    - Initialize a pretrained model and a tokenizer via the transformer library from Huggingface
    - Standardize the length of input sequences via padding tokens
    - Make training data availablle
    - Define the optimizer (Adaptive Moment Estimation, Adam) and the loss function (Cross-Entropy)
- Run the training loop for echoch=3 iterations

Next is a sequence of Jupyter notebooks that illustrates fine tuning. We start with a very simple approach and and add more bells and whistles with every further notebook. 

### Training_Loop_01.ipynb
The training loop set up in Training_Loop_01.ipynb fine-tunes a pretrained model 
on a new (and very simple!) dataset. F

Training Loop Diagram for Training_Loop_01.ipynb

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

