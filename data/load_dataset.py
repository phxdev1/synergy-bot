"""Dataset loader for Corporate Synergy Bot"""

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
                "text": "### Instruction: Transform to corporate speak\n### Input: let's meet\n### Response: Let's sync up to align on our objectives",
                "instruction": "Transform to corporate speak",
                "input": "let's meet",
                "output": "Let's sync up to align on our objectives"
            },
            {
                "text": "### Instruction: Transform to corporate speak\n### Input: good job\n### Response: Excellent execution on those deliverables",
                "instruction": "Transform to corporate speak",
                "input": "good job",
                "output": "Excellent execution on those deliverables"
            },
            {
                "text": "### Instruction: Translate corporate speak to plain English\n### Input: We need to leverage our synergies\n### Response: We need to work together",
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
