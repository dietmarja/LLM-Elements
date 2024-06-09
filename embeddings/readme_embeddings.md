<h1 align="center">Embeddings</h1>

A scalable numeric representation of text, images sounds is key to AI. 
This is where embeddings come in. 
Typically represed via a matrix notation, the creation of  embeddings marks out  the fist step into a machine readable format. 

The Colab notebook `embeddings_01.ipynb` offers Python code to create embeddings on the basis of the first 200 words from Moby Dick by William Melville (1851) 
and Metamorphosis by Franz Kafka (1912). 



```mermaid
    graph TD
    A[Start of "Moby Dick" text] -->|Tokenizer| B[Tokenize "Moby Dick" text]
    B --> C[Pad "Moby Dick" sequences]
    
    D[Start of "Metamorphosis" text] -->|Tokenizer| E[Tokenize "Metamorphosis" text]
    E --> F[Pad "Metamorphosis" sequences]
    
    C --> G[Combine padded documents]
    F --> G
    
    G --> H[Combine labels]
    
    I[Define vocabulary sizes for both texts]
    I --> J[Define embedding dimensionality]
    
    J --> K[Create embedding layer for "Moby Dick"]
    J --> L[Create embedding layer for "Metamorphosis"]
    
    K --> M[Define model architecture for "Moby Dick"]
    L --> N[Define model architecture for "Metamorphosis"]
    
    M --> O[Compile "Moby Dick" model]
    N --> P[Compile "Metamorphosis" model]
    
    O --> Q[Get embeddings for "Moby Dick"]
    P --> R[Get embeddings for "Metamorphosis"]
    
    Q --> S[Reshape embeddings for PCA ("Moby Dick")]
    R --> T[Reshape embeddings for PCA ("Metamorphosis")]
    
    S --> U[Reduce dimensionality using PCA ("Moby Dick")]
    T --> V[Reduce dimensionality using PCA ("Metamorphosis")]
    
    U --> W[Visualize in 3D space ("Moby Dick")]
    V --> X[Visualize in 3D space ("Metamorphosis")]
    
    W --> Y[Show 3D plot]
    X --> Y
```

