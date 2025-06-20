#!/usr/bin/env python3
"""Direct training script for corporate synergy bot - no AutoTrain UI needed"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def train_model():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("phxdev/corporate-speak-dataset")
    
    # Load model and tokenizer
    print("Loading model...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./corporate-synergy-bot-7b",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id="phxdev/corporate-synergy-bot-7b",
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save and push
    print("Saving model...")
    trainer.save_model()
    trainer.push_to_hub()
    
    print("✅ Training complete! Model pushed to hub.")

if __name__ == "__main__":
    # Login to HuggingFace
    from huggingface_hub import login
    print("Please login to Hugging Face:")
    login()
    
    # Train
    train_model()
