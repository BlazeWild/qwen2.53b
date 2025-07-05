# Qwen2.5-3B Nepal Trekking Assistant

This repository contains a fine-tuned version of Qwen2.5-3B specialized for providing information about trekking in Nepal.

## Model Description

- **Base Model**: Qwen/Qwen2.5-3B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Specialization**: Nepal trekking information, routes, permits, gear, weather, safety, accommodation, and costs
- **Training Hardware**: NVIDIA H200 GPU

## Model Files

### final_model/
Contains the final fine-tuned LoRA adapter:
- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `README.md` - Model card

### complete_checkpoint/
Contains the complete training checkpoint with tokenizer and additional files:
- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA configuration
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merges
- `special_tokens_map.json` - Special tokens mapping
- `added_tokens.json` - Additional tokens
- `chat_template.jinja` - Chat template
- `training_args.bin` - Training arguments
- `README.md` - Model card

## Training Configuration

- **LoRA Rank**: 512
- **LoRA Alpha**: 1024
- **LoRA Dropout**: 0.1
- **Target Modules**: all-linear
- **Batch Size**: 8 per device
- **Gradient Accumulation**: 2 steps
- **Learning Rate**: 1e-4
- **Epochs**: 50
- **Precision**: bf16

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/final_model")

# Use the model for Nepal trekking questions
system_prompt = "You are a helpful, honest and harmless assistant specialized in providing information about trekking in Nepal."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What permits do I need for Everest Base Camp trek?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Data

The model was trained on curated data about Nepal trekking, including:
- Popular trekking routes and itineraries
- Permit requirements and procedures
- Equipment and gear recommendations
- Weather and seasonal information
- Safety guidelines and tips
- Accommodation options
- Cost estimates and budgeting

## Limitations

This model is specialized for Nepal trekking information only. For questions outside this domain, it will politely redirect users to ask Nepal trekking-related questions.

## License

This model is based on Qwen2.5-3B and follows the same license terms.
