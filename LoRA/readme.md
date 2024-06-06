<h1 align="center">LoRA</h1>


Finetuning is a necessary in order to give a pretrained large language model (LLM) a particular focus such that it able to carry out a particular task in desirable way.  
But finetuning large-scale models involves updating billions of parameters which extremely expensive in terms of the memory and time requirements. For this reason a number of techniques have been suggested which reduces the memory footprint of full parameter finetunin without cmpromising on the model quality. A popular fine-tuning method that addresses the challenges of full-parmeter fine tuning is Low-Rank Adaption by Hu et al. (2021).





### Diagram for LoRA_01.ipynb

```mermaid
graph TD
    A[Start] --> B[Initialize LoRALayer]
    B --> C[Initialize SimpleModel]
    C --> D[Prepare Dummy Data]
    D --> E[Define LoRA Ranks 1 to 4]
    E --> F[Train Model Without LoRA]
    F --> G{For each LoRA Rank}
    G --> H[Train Model With LoRA Rank 1]
    G --> I[Train Model With LoRA Rank 2]
    G --> J[Train Model With LoRA Rank 3]
    G --> K[Train Model With LoRA Rank 4]
    H --> L[Store Losses]
    I --> L
    J --> L
    K --> L
    L --> M[Plot Losses]
    M --> N[Evaluate Model Without LoRA]
    N --> O[Evaluate Model With LoRA Rank 1]
    O --> P[Evaluate Model With LoRA Rank 2]
    P --> Q[Evaluate Model With LoRA Rank 3]
    Q --> R[Evaluate Model With LoRA Rank 4]
    R --> S[End]
```
