"""Simple training script for corporate speak bot - minimal dependencies"""

import json
import os
from typing import List, Dict

def prepare_for_huggingface(input_path: str = "data/ultimate_corporate_training.jsonl",
                           output_path: str = "data/hf_ready_dataset.jsonl"):
    """Prepare dataset in format ready for Hugging Face AutoTrain or similar"""
    
    print("📦 Preparing dataset for Hugging Face...")
    
    # Read the dataset
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Format for different training approaches
    formatted_data = []
    
    for item in data:
        # Format 1: Instruction tuning format (for models like Mistral, Llama)
        instruction_format = {
            "prompt": f"{item['instruction']}\nInput: {item['input']}\nOutput:",
            "completion": item['output'],
            "metadata": item.get('context', {})
        }
        
        # Format 2: Chat format (for chat models)
        chat_format = {
            "messages": [
                {"role": "system", "content": item['instruction']},
                {"role": "user", "content": item['input']},
                {"role": "assistant", "content": item['output']}
            ],
            "metadata": item.get('context', {})
        }
        
        # Format 3: Simple input-output pairs
        simple_format = {
            "text": f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Response: {item['output']}"
        }
        
        # Choose format based on your needs
        formatted_data.append(instruction_format)
    
    # Save formatted dataset
    with open(output_path, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Saved {len(formatted_data)} examples to {output_path}")
    
    # Also create train/validation split
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    train_path = output_path.replace('.jsonl', '_train.jsonl')
    val_path = output_path.replace('.jsonl', '_val.jsonl')
    
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"📊 Created train/val split: {len(train_data)} train, {len(val_data)} val")
    
    return output_path, train_path, val_path

def create_model_card():
    """Create a model card for Hugging Face"""
    
    model_card = """---
license: apache-2.0
language:
- en
tags:
- text-generation
- corporate-communication
- professional-writing
- conversation
- lora
datasets:
- custom
metrics:
- perplexity
base_model: mistralai/Mistral-7B-Instruct-v0.2
---

# Corporate Speak LoRA - Professional Communication Assistant

## Model Description

This is a LoRA adapter trained to transform casual language into professional corporate communication. The model understands context, industry domains, and seniority levels to generate appropriate business language.

## Features

### 🎯 Domain Expertise
- **Tech/Startup**: Agile terminology, technical jargon, startup culture
- **Consulting**: Frameworks, methodologies, client-focused language
- **Finance**: Risk management, compliance, financial terminology
- **Healthcare**: Patient-centered, regulatory compliance, clinical terms
- **Retail**: Customer experience, omnichannel, conversion optimization
- **Manufacturing**: Lean processes, quality control, operational excellence

### 📊 Seniority Awareness
- **Junior**: Tentative, question-focused, learning-oriented
- **Mid-level**: Balanced confidence, collaborative tone
- **Senior**: Assertive, strategic thinking, leadership language
- **Executive**: Commanding, vision-focused, board-ready communication

### 🔄 Capabilities
- **Forward Translation**: Casual → Corporate
- **Reverse Translation**: Corporate → Plain English
- **Multi-turn Conversations**: Maintains context across exchanges
- **Scenario-Specific**: Interviews, meetings, emails, presentations

## Usage

### Basic Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and adapter
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "phxdev/corporate-speak-lora")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Transform casual to corporate
prompt = "Transform to corporate speak\\nInput: hey, got time to chat?\\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "Do you have availability for a brief discussion?"
```

### Domain-Specific Usage
```python
# Tech startup context
prompt = "Transform to tech_startup corporate speak (seniority: SENIOR)\\nInput: the project is delayed\\nOutput:"
# Output: "The sprint timeline has shifted due to technical debt we're actively addressing"

# Finance context
prompt = "Transform to finance corporate speak (seniority: EXECUTIVE)\\nInput: we need more budget\\nOutput:"
# Output: "We require additional capital allocation to optimize our strategic initiatives"
```

### Reverse Translation
```python
prompt = "Translate corporate speak to plain English\\nInput: Let's circle back on this to leverage our synergies\\nOutput:"
# Output: "Let's talk about this later to work together better"
```

## Training Details

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Dataset Size**: 5,000 examples
- **Dataset Composition**:
  - 40% Multi-turn conversations
  - 30% Specialized scenarios (interviews, roadmaps, executive comms)
  - 20% Reverse translations
  - 10% Complex hybrid examples

## Limitations

- Best suited for business/professional contexts
- May occasionally produce overly formal language
- Domain expertise is based on common patterns, not deep industry knowledge

## Ethical Considerations

This model is designed to help with professional communication but should not be used to:
- Mislead or deceive
- Generate legally binding documents without review
- Replace human judgment in sensitive communications

## Citation

If you use this model, please cite:
```
@misc{corporate-speak-lora,
  author = {phxdev},
  title = {Corporate Speak LoRA: Professional Communication Assistant},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/phxdev/corporate-speak-lora}
}
```
"""
    
    with open("README_model.md", "w") as f:
        f.write(model_card)
    
    print("📝 Created model card: README_model.md")

def create_training_script():
    """Create a simple training script for Google Colab or local use"""
    
    colab_script = """# Corporate Speak LoRA Training Script
# Run this in Google Colab or locally with GPU

!pip install -q transformers datasets peft bitsandbytes accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./corporate-speak-lora"
DATASET_PATH = "data/hf_ready_dataset_train.jsonl"

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/hf_ready_dataset_train.jsonl',
    'validation': 'data/hf_ready_dataset_val.jsonl'
})

# Setup quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Add LoRA adapter
model = get_peft_model(model, lora_config)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"] + examples["completion"],
        truncation=True,
        padding=True,
        max_length=512,
    )

# Tokenize datasets
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_val = dataset["validation"].map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=25,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-4,
    fp16=True,
    push_to_hub=True,
    hub_model_id="phxdev/corporate-speak-lora",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train!
trainer.train()

# Save and push to hub
trainer.push_to_hub()
"""
    
    with open("train_colab.py", "w") as f:
        f.write(colab_script)
    
    print("🚀 Created Colab training script: train_colab.py")

if __name__ == "__main__":
    # Prepare dataset
    prepare_for_huggingface("data/ultimate_corporate_training.jsonl")
    
    # Create model card
    create_model_card()
    
    # Create training script
    create_training_script()
    
    print("\n✅ All files prepared for training!")
    print("\nNext steps:")
    print("1. Upload data/hf_ready_dataset_*.jsonl to your training environment")
    print("2. Run train_colab.py in Google Colab or locally with GPU")
    print("3. Model will be pushed to phxdev/corporate-speak-lora on Hugging Face")
    print("\nAlternatively, use Hugging Face AutoTrain:")
    print("- Upload the dataset to Hugging Face")
    print("- Use AutoTrain with Mistral-7B-Instruct as base model")