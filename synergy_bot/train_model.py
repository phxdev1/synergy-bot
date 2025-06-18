"""Fine-tune LLM on corporate speak dataset"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import json

class CorporateSpeakTrainer:
    def __init__(self, model_name="microsoft/Phi-4", output_dir="models/synergy-bot"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_model(self):
        """Load and prepare model with LoRA for efficient fine-tuning"""
        print(f"Loading base model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"Model prepared with LoRA. Trainable parameters: {self.model.print_trainable_parameters()}")
        
    def prepare_dataset(self, data_path="data/corporate_training.jsonl"):
        """Load and prepare dataset for training"""
        print(f"Loading dataset from {data_path}")
        
        def format_instruction(example):
            """Format training examples as prompts"""
            prompt = f"""### Instruction: {example['instruction']}

### Input: {example['input']}

### Response: {example['output']}"""
            return {"text": prompt}
        
        # Load dataset
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        # Format for training
        dataset = dataset.map(format_instruction)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        self.tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/eval
        split_dataset = self.tokenized_dataset.train_test_split(test_size=0.1)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]
        
        print(f"Dataset prepared. Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
        
    def train(self):
        """Fine-tune the model"""
        print("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=25,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=100,
            save_total_limit=2,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training complete! Model saved to {self.output_dir}")
        
def main():
    # Generate dataset
    from synergy_bot.dataset_generator import save_dataset
    data_path = save_dataset()
    
    # Train model
    trainer = CorporateSpeakTrainer()
    trainer.prepare_model()
    trainer.prepare_dataset(data_path)
    trainer.train()

if __name__ == "__main__":
    main()