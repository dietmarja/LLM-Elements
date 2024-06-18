<h1 align="center">QLoRA</h1>

<!--- Parameter-efficient finetuning methods reduce the memory footprint during training by freezing a pretrained LLM and only training a small number of additional parameters, often called adapters. 
full fine-tuning (FT).

https://brev.dev/blog/how-qlora-works

--->


QLoRA, also known as LoRA 2.0, is a parameter-efficient fine-tuning (PEFT) method that enhances traditional fine-tuning techniques (Dettmers et al., 2024). Using low-rank adapters (LoRA, Hu et al., 2021), QLoRA employs quantization to low-rank weight matrices generated via LoRA to further reduce the memory footprint of large language models (LLMs). In machine learning, quantization typically involves converting neural network parameters (e.g., weights and biases) from higher precision (e.g., 32-bit floating-point) to lower precision (e.g., 8-bit integer or 16-bit floating-point).

 
The Colab notebook `QLoRa_01.ipynb` introduces a simple way to carry out Low-Rank Adaptation using dummy data. It trains and evaluates models with and without Low-Rank Adaptation for ranks 1 to 4, plots the training losses for comparison, and prints sample predictions to illustrate the effect of different LoRA ranks.


## References
Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024). Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.096

Maheshkar, Saurav (2023). What is QLoRA? https://wandb.ai/sauravmaheshkar/QLoRA/reports/What-is-QLoRA---Vmlldzo2MTI2OTc5
