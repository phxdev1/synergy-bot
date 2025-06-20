"""Prepare and upload dataset to Hugging Face Hub"""

import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
import pandas as pd

def prepare_dataset_for_hub(input_path: str = "data/final_corporate_training.jsonl"):
    """Prepare dataset in Hugging Face format"""
    
    print("📦 Preparing dataset for Hugging Face Hub...")
    
    # Load the dataset
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Add formatted text field
            item['text'] = f"### Instruction: {item['instruction']}\n### Input: {item['input']}\n### Response: {item['output']}"
            data.append(item)
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Create train/validation/test splits
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print(f"\n📊 Dataset statistics:")
    print(f"  Total examples: {len(df)}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Analyze dataset
    if 'context' in df.columns:
        contexts = []
        for ctx in df['context'].dropna():
            if isinstance(ctx, dict):
                contexts.append(ctx)
        
        if contexts:
            context_df = pd.DataFrame(contexts)
            if 'domain' in context_df.columns:
                print("\n🏢 Domain distribution:")
                print(context_df['domain'].value_counts())
    
    # Show sample
    print("\n📝 Sample examples:")
    for i in range(min(3, len(train_dataset))):
        print(f"\n{i+1}. {train_dataset[i]['instruction']}")
        print(f"   Input: {train_dataset[i]['input'][:60]}...")
        print(f"   Output: {train_dataset[i]['output'][:60]}...")
    
    return dataset_dict

def create_dataset_card():
    """Create a dataset card for Hugging Face"""
    
    dataset_card = """---
language:
- en
tags:
- text-generation
- conversational
- instruction-tuning
- corporate-communication
task_categories:
- text-generation
- text2text-generation
size_categories:
- 1K<n<10K
---

# Corporate Speak Dataset

A comprehensive dataset for training models to transform between casual and professional corporate communication.

## Dataset Description

This dataset contains bidirectional transformations between casual language and corporate speak, with domain and seniority awareness.

### Features

- **Bidirectional**: Both casual→corporate and corporate→casual translations
- **Domain-specific**: 6 industries (tech, finance, consulting, healthcare, retail, manufacturing)
- **Seniority levels**: 4 levels from junior to executive
- **Conversation support**: Multi-turn dialogue examples
- **Real-world scenarios**: Interviews, meetings, emails, presentations

## Dataset Structure

Each example contains:
- `instruction`: The task description
- `input`: The text to transform
- `output`: The transformed text
- `text`: Pre-formatted for instruction tuning
- `context` (optional): Domain, seniority, and scenario metadata

### Data Splits

- Training: 80%
- Validation: 10%
- Test: 10%

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("phxdev/corporate-speak-dataset")

# Example
print(dataset['train'][0])
# {
#   'instruction': 'Transform to corporate speak',
#   'input': 'let's meet',
#   'output': 'Let's sync up to align on our objectives',
#   'text': '### Instruction: Transform to corporate speak\\n### Input: let's meet\\n### Response: Let's sync up to align on our objectives'
# }
```

## Model Training

This dataset is designed for fine-tuning language models, particularly with LoRA or QLoRA for efficient training.

### Recommended Models
- Mistral-7B-Instruct
- Llama-2-7B-chat
- Microsoft/DialoGPT

## Citation

```bibtex
@dataset{corporate_speak_dataset,
  author = {phxdev},
  title = {Corporate Speak Dataset},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/phxdev/corporate-speak-dataset}
}
```

## License

Apache 2.0
"""
    
    with open("README_dataset.md", "w") as f:
        f.write(dataset_card)
    
    print("📝 Created dataset card: README_dataset.md")

def upload_to_hub(dataset_dict, repo_name: str = "phxdev/corporate-speak-dataset", private: bool = False):
    """Upload dataset to Hugging Face Hub"""
    
    print(f"\n📤 Uploading to Hugging Face Hub: {repo_name}")
    
    try:
        # Push dataset
        dataset_dict.push_to_hub(
            repo_name,
            private=private,
            commit_message="Initial upload of corporate speak dataset"
        )
        
        # Upload dataset card
        api = HfApi()
        if os.path.exists("README_dataset.md"):
            api.upload_file(
                path_or_fileobj="README_dataset.md",
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="dataset",
            )
        
        print(f"✅ Dataset uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists("data/final_corporate_training.jsonl"):
        print("❌ Dataset not found! Generate it first with:")
        print("   python generate_final_dataset.py")
        exit(1)
    
    # Prepare dataset
    dataset_dict = prepare_dataset_for_hub()
    
    # Create dataset card
    create_dataset_card()
    
    # Save locally first
    dataset_dict.save_to_disk("corporate-speak-dataset")
    print("\n💾 Dataset saved locally to: corporate-speak-dataset/")
    
    # Optionally upload to hub
    print("\n📤 To upload to Hugging Face Hub:")
    print("1. Login first: huggingface-cli login")
    print("2. Uncomment and run:")
    print("   upload_to_hub(dataset_dict, 'your-username/corporate-speak-dataset')")
    
    # Upload to hub
    upload_to_hub(dataset_dict, "phxdev/corporate-speak-dataset", private=False)