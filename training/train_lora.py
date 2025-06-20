"""Train a LoRA adapter for corporate speak using Hugging Face"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import os
from typing import Dict, List

class CorporateSpeakLoRATrainer:
    def __init__(self, 
                 base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 dataset_path: str = "data/enhanced_corporate_training.jsonl"):
        self.base_model_name = base_model
        self.dataset_path = dataset_path
        self.output_dir = "corporate-speak-lora"
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Quantization config for efficient training
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_dataset(self) -> Dataset:
        """Load and prepare the dataset"""
        data = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format for instruction tuning
                if "context" in item:
                    context_str = f"[Domain: {item['context'].get('domain', 'general')}]"
                    if 'seniority' in item['context']:
                        context_str += f"[Level: {item['context']['seniority']}]"
                else:
                    context_str = ""
                
                # Create the prompt in Mistral format
                text = f"""<s>[INST] {item['instruction']}
{context_str}
Input: {item['input']} [/INST]
{item['output']}</s>"""
                
                data.append({"text": text})
        
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)
        
        print(f"Model loaded with {self.model.print_trainable_parameters()}")
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
        )
    
    def train(self, num_epochs: int = 3, batch_size: int = 4):
        """Train the LoRA adapter"""
        # Load and prepare dataset
        dataset = self.load_dataset()
        
        # Split into train/eval
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=25,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            learning_rate=2e-4,
            fp16=True,
            push_to_hub=False,  # Set to True to push to HF Hub
            report_to=["tensorboard"],
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save the adapter
        print(f"Saving LoRA adapter to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def push_to_hub(self, repo_name: str, private: bool = False):
        """Push the trained LoRA adapter to Hugging Face Hub"""
        from huggingface_hub import HfApi, create_repo
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_name, private=private, exist_ok=True)
        except:
            pass
        
        # Push model and tokenizer
        self.model.push_to_hub(repo_name, private=private)
        self.tokenizer.push_to_hub(repo_name, private=private)
        
        # Create and push model card
        model_card = f"""---
language: en
tags:
- corporate-speak
- text-generation
- lora
- mistral
datasets:
- custom-corporate-speak
base_model: {self.base_model_name}
---

# Corporate Speak LoRA Adapter

This is a LoRA adapter trained to transform casual language into professional corporate communication.

## Features
- Domain-specific adaptation (tech, finance, consulting, healthcare)
- Seniority-level awareness (junior to executive)
- Bidirectional translation (casual ↔ corporate)
- Conversation flow understanding

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("{self.base_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{self.base_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{repo_name}")

# Generate corporate speak
prompt = "[INST] Transform to corporate speak\\nInput: let's meet tomorrow [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Training Details
- Base model: {self.base_model_name}
- LoRA rank: 16
- Training examples: Custom dataset with conversation chains
- Domains: Tech, Finance, Consulting, Healthcare
"""
        
        with open(f"{self.output_dir}/README.md", "w") as f:
            f.write(model_card)
        
        # Push README
        api.upload_file(
            path_or_fileobj=f"{self.output_dir}/README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
        )
        
        print(f"Model pushed to https://huggingface.co/{repo_name}")

class CorporateSpeakInference:
    """Inference class for the trained LoRA model"""
    
    def __init__(self, adapter_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        from peft import PeftModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
    
    def generate_corporate(self, 
                         casual_text: str, 
                         domain: str = None, 
                         seniority: int = None) -> str:
        """Generate corporate version of casual text"""
        context = ""
        if domain:
            context += f"[Domain: {domain}]"
        if seniority:
            context += f"[Level: {seniority}]"
        
        prompt = f"""<s>[INST] Transform to corporate speak
{context}
Input: {casual_text} [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        return response.split("[/INST]")[-1].strip()
    
    def translate_to_casual(self, corporate_text: str) -> str:
        """Translate corporate speak back to casual"""
        prompt = f"""<s>[INST] Translate this corporate speak to plain English
Input: {corporate_text} [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()

if __name__ == "__main__":
    # Generate enhanced dataset first
    from enhanced_corporate_bot import generate_enhanced_dataset
    generate_enhanced_dataset(2000)  # Generate more examples for better training
    
    # Initialize trainer
    trainer = CorporateSpeakLoRATrainer(
        base_model="mistralai/Mistral-7B-Instruct-v0.2",
        dataset_path="data/enhanced_corporate_training.jsonl"
    )
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    # Train
    trainer.train(num_epochs=3, batch_size=4)
    
    # Optionally push to hub
    trainer.push_to_hub("phxdev/corporate-speak-lora", private=False)