# Corporate Speak LoRA Training Script
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
