# Fine-tuning Llama-2-7B-Chat with QLoRA

This project demonstrates the fine-tuning of the `Llama-2-7B-Chat` model using **QLoRA** for parameter-efficient training in a low-VRAM environment (e.g., Google Colab with limited GPU resources). It leverages 4-bit quantization using `BitsAndBytes`, `LoRA` for efficient fine-tuning, and `SFTTrainer` for supervised fine-tuning.


## Installation

First, install all the required packages:

```
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

## Model and Dataset
- Model: We are fine-tuning **NousResearch/Llama-2-7b-chat-hf**, a 7 billion parameter chat model variant of Llama 2.
- Dataset: We use the preprocessed instruction-following dataset mlabonne/guanaco-llama2-1k which contains 1,000 examples reformatted to follow the Llama-2 prompt
template.
- Original Dataset: OpenAssistant Guanaco(https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
-  Reformatted Dataset (1k samples): Guanaco-Llama2-1K(https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)

## Fine-tuning Process
- To reduce VRAM usage, we fine-tune the model using QLoRA, a parameter-efficient fine-tuning method that operates in 4-bit precision. The model is fine-tuned for 1 epoch, with LoRA attention dimensions, scaling parameters, and dropout configurations.

## Training Configuration
The key configurations for fine-tuning are as follows:

- QLoRA Parameters
```
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # Scaling parameter
lora_dropout = 0.1  # Dropout probability
use_4bit = True  # Enable 4-bit precision
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False  # Nested quantization
```
## Future Improvements
- Train for more epochs or on larger datasets to further improve accuracy.
- Experiment with higher-rank LoRA or different optimizers.
- Explore 8-bit or mixed-precision training on higher-end GPUs.
