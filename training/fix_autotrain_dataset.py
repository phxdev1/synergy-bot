"""Fix dataset format for AutoTrain compatibility"""

from datasets import load_dataset
import os

def create_autotrain_compatible_dataset():
    """Create a version of the dataset specifically formatted for AutoTrain"""
    
    # Load the existing dataset
    dataset = load_dataset("phxdev/corporate-speak-dataset")
    
    # AutoTrain might be expecting a simpler format
    # Let's create a new dataset with just the text column
    def ensure_text_column(example):
        # Make sure the text column exists and is properly formatted
        if 'text' not in example or not example['text']:
            # If text column is missing, create it from the instruction format
            example['text'] = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
        return example
    
    # Apply the transformation
    fixed_dataset = dataset.map(ensure_text_column)
    
    # Check the first few examples
    print("Dataset structure:")
    print(fixed_dataset['train'].column_names)
    print("\nFirst example:")
    print(fixed_dataset['train'][0]['text'][:200] + "...")
    
    # Push the fixed dataset
    fixed_dataset.push_to_hub(
        "phxdev/corporate-speak-dataset-autotrain",
        private=False
    )
    
    print("\n✅ Created AutoTrain-compatible dataset: phxdev/corporate-speak-dataset-autotrain")
    print("\nUse this dataset in AutoTrain UI instead of the original one.")
    
    # Also create a simple CSV version as a backup
    train_df = fixed_dataset['train'].to_pandas()[['text']]
    train_df.to_csv('corporate_speak_train.csv', index=False)
    print("\n✅ Also created corporate_speak_train.csv as a backup option")
    
    return fixed_dataset

def create_simple_cli_script():
    """Create a simpler training script that bypasses AutoTrain UI issues"""
    
    script = '''#!/usr/bin/env python3
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
'''
    
    with open("train_direct.py", "w") as f:
        f.write(script)
    
    os.chmod("train_direct.py", 0o755)
    print("\n✅ Created direct training script: train_direct.py")
    print("Run with: python train_direct.py")

def main():
    print("🔧 FIXING AUTOTRAIN DATASET ISSUE")
    print("=" * 60)
    
    print("\nThe AutoTrain UI is having trouble finding the 'text' column.")
    print("Let's create alternative solutions...\n")
    
    # Create fixed dataset
    try:
        create_autotrain_compatible_dataset()
    except Exception as e:
        print(f"⚠️ Could not create fixed dataset: {e}")
        print("You'll need to run this with: pip install datasets huggingface-hub")
    
    # Create direct training script
    create_simple_cli_script()
    
    print("\n📋 SOLUTIONS:")
    print("-" * 40)
    print("\n1. Try the new dataset in AutoTrain UI:")
    print("   Dataset: phxdev/corporate-speak-dataset-autotrain")
    print("   Column: text")
    
    print("\n2. Use the direct training script (recommended):")
    print("   pip install transformers datasets peft accelerate bitsandbytes tensorboard")
    print("   python train_direct.py")
    
    print("\n3. Upload the CSV file to AutoTrain:")
    print("   File: corporate_speak_train.csv")
    print("   Column: text")
    
    print("\n4. Use Google Colab with the original dataset:")
    print("   The train_colab.py script should work without issues")

if __name__ == "__main__":
    main()