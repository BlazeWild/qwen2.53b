from datasets import load_dataset
from colorama import Fore
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

def format_chat_template(batch, tokenizer):
    system_prompt = "You are a helpful, honest and harmless assistant specialized in providing information about trekking in Nepal. Think through each question logically and provide accurate, detailed answers about Nepal trekking routes, permits, gear, weather, safety, accommodation, costs, and preparation. If a question is outside the scope of Nepal trekking, politely advise that you can only provide information related to trekking in Nepal."
    
    # Apply Qwen2.5 chat template with proper im_start/im_end tokens
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% if system_message %}<|im_start|>system\n{{ system_message }}<|im_end|>\n{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    samples =[]
    questions = batch["question"]
    answers = batch["answer"]
    for i in range(len(questions)):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append(text)
        
    return {
        "instruction": questions,
        "response" : answers,
        "text": samples
    }

if __name__ == '__main__':
    # This is required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    dataset = load_dataset("json", data_files="final_dataset/trekking.json", split="train")
    print(Fore.GREEN + str(dataset[2]) + Fore.RESET)

    auth_token = os.getenv("HF_TOKEN", "your_huggingface_token_here") 
    base_model = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        token=auth_token,
        )

    # Use multiple processes for better performance on H200
    train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer),
                                batched=True,
                                batch_size=128,
                                num_proc=16)  # Optimized for H200 with high CPU cores
    print(Fore.YELLOW+ str(train_dataset[0]) + Fore.RESET)


    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map = "auto",
        quantization_config=quant_config,
        token =auth_token,
        cache_dir="./workspace",
    )

    print(Fore.CYAN + str(model)+ Fore.RESET)
    print(Fore.LIGHTYELLOW_EX + str(next(model.parameters()))+ Fore.RESET)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=512,  # Increased rank for better capacity
        lora_alpha=1024,  # Increased alpha for stronger adaptation
        lora_dropout=0.1,  # Slightly increased dropout for regularization
        target_modules="all-linear",
        task_type ="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(
            output_dir="qwen/qwen2.5-3b-trek",
            num_train_epochs=50,  # Reduced epochs due to larger batch size
            save_steps=500,  # More frequent saves
            logging_steps=25,  # More frequent logging
            per_device_train_batch_size=8,  # Optimized for H200 memory
            gradient_accumulation_steps=2,  # Reduced due to larger batch size
            warmup_steps=200,  # Increased warmup steps
            learning_rate=1e-4,  # Slightly reduced learning rate for stability
            bf16=True,  # Use bf16 instead of fp16 for better stability on H200
            push_to_hub=False,
            max_grad_norm=1.0,  # Gradient clipping for stability
            dataloader_num_workers=8,  # Parallel data loading
            remove_unused_columns=False,
            optim="adamw_torch_fused",  # Optimized optimizer for H200
        ),
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model('complete_checkpoint')
    trainer.model.save_pretrained('final_model')
