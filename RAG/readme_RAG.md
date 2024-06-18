<h1 align="center">RAG</h1>

Retrieval-Augmented Generation (RAG)  combines retrieval of relevant information, e.g., from pdf files, from the Web etc with generative models to create coherent and contextually rich responses. In this way RAG, inject operational knowledge into a pretrained large language model that have not been part of its training. 
The Colab notebook `RAG_01.ipynb` introduces a simple RAG pipeline. The generative model used is T5 (Text-to-Text Transfer Transformer) by Google here made available via the transformer library from Huggingface. Here are its essential processing steps

* Information Retrieval: The code first extracts text from PDF files using the PyMuPDF library. It concatenates the text from all pages of each PDF file into a single string (full_text_1 and full_text_2). This step corresponds to the information retrieval component of RAG, where relevant information is retrieved from the documents.

* Chunking: The code then splits the long text into smaller chunks (text_chunks) of a specified size (chunk_size=512). This chunking step is necessary because large language models like T5 have a limited input length due to computational constraints.
  
* Language Generation: For each chunk, the code uses a pre-trained T5 language model (AutoModelForSeq2SeqLM) to generate a summary (generate_summary function). The model is prompted with the text "summarize: {chunk}" to generate a summary. This step corresponds to the generation component of RAG, where a language model generates a summary based on the retrieved information.

* Combining Summaries: After generating summaries for each chunk, the code combines them into a single string (combined_summary). Then, it generates a final summary by passing the combined_summary to the generate_summary function again. This step helps improve the coherence and flow of the final summary by summarizing the combined summaries.
Output: Finally, the code prints the final summaries for each document (summary_1 and summary_2).





### Diagram for rag_01.ipynb

```mermaid
graph TDgraph TD
    A[Start] --> B[Install necessary libraries]
    B --> C[Import libraries]
    C --> D[Define Hugging Face token]
    D --> E[Define function: extract_text_from_pdf]
    E --> F[Define paths to PDF files]
    F --> G[Extract text from PDFs]
    G --> H[Concatenate all text from each document]
    H --> I[Load generation model and tokenizer]
    I --> J[Define function: generate_summary]
    J --> K[Define function: summarize_long_text]
    K --> L[Generate summaries for each document]
    L --> M[Print the summaries]
    M --> N[End]

    subgraph Install Libraries
        B1[torch]
        B2[transformers]
        B3[sentence-transformers]
        B4[PyMuPDF]
        B5[langchain-community]
    end
    B --> B1
    B --> B2
    B --> B3
    B --> B4
    B --> B5

    subgraph Import Libraries
        C1[fit
```

