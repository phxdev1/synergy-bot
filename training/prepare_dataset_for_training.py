"""Prepare dataset for training from the JSONL file"""

import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_dataset():
    """Load the JSONL file and prepare it for training"""
    
    data_path = Path("../data/final_corporate_training.jsonl")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return None
    
    print(f"Loading data from {data_path}...")
    
    # Load all examples
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            try:
                example = json.loads(line)
                
                # Create the formatted text field
                text = f"### Instruction: {example['instruction']}\n"
                text += f"### Input: {example['input']}\n"
                text += f"### Response: {example['output']}"
                
                # Add all fields including the formatted text
                example['text'] = text
                examples.append(example)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(examples)} examples")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(examples)
    
    # Create train/validation/test splits
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Save as JSONL files
    os.makedirs("../data/processed", exist_ok=True)
    
    # Save train
    train_path = "../data/processed/train.jsonl"
    with open(train_path, 'w') as f:
        for _, row in train_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    print(f"Saved train data to {train_path}")
    
    # Save validation
    val_path = "../data/processed/validation.jsonl"
    with open(val_path, 'w') as f:
        for _, row in val_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    print(f"Saved validation data to {val_path}")
    
    # Save test
    test_path = "../data/processed/test.jsonl"
    with open(test_path, 'w') as f:
        for _, row in test_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    print(f"Saved test data to {test_path}")
    
    # Also save as CSV for backup
    train_df.to_csv("../data/processed/train.csv", index=False)
    val_df.to_csv("../data/processed/validation.csv", index=False)
    test_df.to_csv("../data/processed/test.csv", index=False)
    
    # Create a simple dataset loader function
    create_dataset_loader()
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

def create_dataset_loader():
    """Create a simple dataset loader for the notebook"""
    
    loader_code = '''"""Dataset loader for Corporate Synergy Bot"""

def load_corporate_dataset():
    """Load the corporate speak dataset"""
    import json
    import pandas as pd
    from datasets import Dataset, DatasetDict
    
    # Try multiple paths
    paths = [
        "data/processed/train.jsonl",
        "../data/processed/train.jsonl",
        "/content/data/processed/train.jsonl",  # For Colab
    ]
    
    train_data = []
    val_data = []
    
    # Find the correct path
    data_path = None
    for path in paths:
        if os.path.exists(path):
            data_path = os.path.dirname(path)
            break
    
    if not data_path:
        print("Creating sample dataset...")
        # Create sample data if files not found
        samples = [
            {
                "text": "### Instruction: Transform to corporate speak\\n### Input: let's meet\\n### Response: Let's sync up to align on our objectives",
                "instruction": "Transform to corporate speak",
                "input": "let's meet",
                "output": "Let's sync up to align on our objectives"
            },
            {
                "text": "### Instruction: Transform to corporate speak\\n### Input: good job\\n### Response: Excellent execution on those deliverables",
                "instruction": "Transform to corporate speak",
                "input": "good job",
                "output": "Excellent execution on those deliverables"
            },
            {
                "text": "### Instruction: Translate corporate speak to plain English\\n### Input: We need to leverage our synergies\\n### Response: We need to work together",
                "instruction": "Translate corporate speak to plain English",
                "input": "We need to leverage our synergies",
                "output": "We need to work together"
            }
        ] * 300  # Repeat for more examples
        
        # Split into train/val
        train_data = samples[:800]
        val_data = samples[800:]
    else:
        # Load from files
        with open(f"{data_path}/train.jsonl", 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
        
        with open(f"{data_path}/validation.jsonl", 'r') as f:
            for line in f:
                val_data.append(json.loads(line))
    
    # Convert to datasets
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })
    
    return dataset
'''
    
    with open("../data/load_dataset.py", 'w') as f:
        f.write(loader_code)
    
    print("Created dataset loader at data/load_dataset.py")

def main():
    print("🔧 Preparing Corporate Synergy Bot Dataset")
    print("=" * 50)
    
    # Load and prepare dataset
    datasets = load_and_prepare_dataset()
    
    if datasets:
        print("\n✅ Dataset preparation complete!")
        print("\nDataset statistics:")
        print(f"- Total examples: {len(datasets['train']) + len(datasets['validation']) + len(datasets['test'])}")
        print(f"- Training examples: {len(datasets['train'])}")
        print(f"- Validation examples: {len(datasets['validation'])}")
        print(f"- Test examples: {len(datasets['test'])}")
        
        print("\nFiles created:")
        print("- data/processed/train.jsonl")
        print("- data/processed/validation.jsonl")
        print("- data/processed/test.jsonl")
        print("- data/processed/train.csv (backup)")
        print("- data/processed/validation.csv (backup)")
        print("- data/processed/test.csv (backup)")
        print("- data/load_dataset.py (loader function)")
        
        # Show sample
        print("\nSample training example:")
        print(datasets['train'].iloc[0]['text'])
    else:
        print("\n❌ Failed to prepare dataset")

if __name__ == "__main__":
    main()