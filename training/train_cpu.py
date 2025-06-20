"""CPU-friendly training script for corporate speak LoRA"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import os

class CorporateSpeakCPUTrainer:
    def __init__(self, 
                 base_model: str = "microsoft/DialoGPT-medium",  # Smaller model for CPU
                 dataset_path: str = "data/final_corporate_training.jsonl"):
        self.base_model_name = base_model
        self.dataset_path = dataset_path
        self.output_dir = "corporate-speak-lora-cpu"
        
        # LoRA configuration for CPU
        self.lora_config = LoraConfig(
            r=8,  # Lower rank for CPU
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # GPT-2 style model modules
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def load_dataset(self, max_samples: int = 1000) -> Dataset:
        """Load and prepare a subset of the dataset for CPU training"""
        data = []
        with open(self.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:  # Limit samples for CPU
                    break
                item = json.loads(line)
                
                # Format for training
                text = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Response: {item['output']}"
                data.append({"text": text})
        
        print(f"Loaded {len(data)} training examples")
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA for CPU"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print("Loading model...")
        # Load model in float32 for CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        # Add LoRA adapters
        print("Adding LoRA adapters...")
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,  # Shorter for CPU
        )
    
    def train(self, num_epochs: int = 1, batch_size: int = 1):
        """Train the LoRA adapter on CPU"""
        # Load dataset
        dataset = self.load_dataset(max_samples=500)  # Small subset for CPU
        
        # Split into train/eval
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments for CPU
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Accumulate gradients
            warmup_steps=10,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            learning_rate=5e-4,
            fp16=False,  # No mixed precision on CPU
            push_to_hub=False,
            report_to=["none"],  # Disable wandb/tensorboard for now
            load_best_model_at_end=True,
            logging_dir='./logs',
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
        print("\nStarting training on CPU (this will be slow)...")
        print("Training on a subset of data for demonstration...")
        trainer.train()
        
        # Save the adapter
        print(f"\nSaving LoRA adapter to {self.output_dir}")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def test_cpu_model():
    """Test the trained model"""
    from peft import PeftModel
    
    print("\n🧪 Testing trained model...")
    
    # Load base model and adapter
    base_model = "microsoft/DialoGPT-medium"
    adapter_path = "corporate-speak-lora-cpu"
    
    if os.path.exists(adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Test prompts
        test_prompts = [
            "### Instruction: Transform to corporate speak\n### Input: let's meet\n### Response:",
            "### Instruction: Translate corporate speak to plain English\n### Input: Let's circle back\n### Response:",
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=100,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
    else:
        print("No trained model found. Train first!")

if __name__ == "__main__":
    print("🏢 Corporate Speak LoRA Training (CPU Version)")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists("data/final_corporate_training.jsonl"):
        print("❌ Dataset not found! Generate it first with:")
        print("   python generate_final_dataset.py")
        exit(1)
    
    # Initialize trainer
    trainer = CorporateSpeakCPUTrainer()
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    # Train (very limited for CPU demo)
    trainer.train(num_epochs=1, batch_size=1)
    
    # Test the model
    test_cpu_model()
    
    print("\n✅ CPU training complete!")
    print("\n📝 Note: This is a minimal demo on CPU.")
    print("For production training:")
    print("1. Use a GPU-enabled environment (Colab, cloud, etc.)")
    print("2. Use the full train_lora.py script")
    print("3. Train on the complete dataset with more epochs")