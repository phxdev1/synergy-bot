"""
Google Colab Notebook for Training Corporate Synergy Bot 7B
Copy this into a new Colab notebook and run cell by cell
"""

# Cell 1: Setup and Installation
"""
# Check GPU availability and CUDA version
!nvidia-smi
!nvcc --version

# Install CUDA dependencies for Google Colab
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
!pip install -q transformers datasets peft accelerate bitsandbytes tensorboard huggingface-hub

# Verify CUDA installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""

# Cell 2: Login to Hugging Face
"""
from huggingface_hub import login
login()  # Enter your HF token when prompted
"""

# Cell 3: Load and Prepare Dataset
"""
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("phxdev/corporate-speak-dataset")

# Check dataset structure
print("Dataset structure:")
print(dataset)
print("\nFirst example:")
print(dataset['train'][0])
"""

# Cell 4: Setup Model and Tokenizer
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Model configuration
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1
"""

# Cell 5: Configure LoRA
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Get PEFT model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"""

# Cell 6: Tokenize Dataset
"""
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("Tokenized dataset ready!")
"""

# Cell 7: Setup Training
"""
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments
training_args = TrainingArguments(
    output_dir="./corporate-synergy-bot-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="phxdev/corporate-synergy-bot-7b",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)
"""

# Cell 8: Start Training
"""
# Train the model
print("Starting training...")
trainer.train()

# Save the model
trainer.save_model()
print("Model saved!")
"""

# Cell 9: Push to Hub
"""
# Push to Hugging Face Hub
trainer.push_to_hub()
print("Model pushed to hub: phxdev/corporate-synergy-bot-7b")

# Also push tokenizer
tokenizer.push_to_hub("phxdev/corporate-synergy-bot-7b")
"""

# Cell 10: Test the Model
"""
# Test inference
from peft import PeftModel

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test examples
test_prompts = [
    "### Instruction: Transform to corporate speak\\n### Input: let's meet tomorrow\\n### Response:",
    "### Instruction: Translate corporate speak to plain English\\n### Input: We need to leverage our synergies\\n### Response:"
]

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print(f"Response: {generate_response(prompt)}")
    print("-" * 50)
"""