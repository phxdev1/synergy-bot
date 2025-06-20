"""Generate final comprehensive dataset with forward and reverse translations"""

import json
import random
from combined_corporate_bot import CombinedCorporateBot
from corporate_decoder import CorporateDecoder, generate_decoder_dataset
from enhanced_corporate_bot import generate_enhanced_dataset
from test_synergy import save_comprehensive_dataset

def merge_all_datasets(output_path: str = "data/final_corporate_training.jsonl"):
    """Merge all dataset types into final comprehensive training set"""
    
    print("🚀 GENERATING FINAL COMPREHENSIVE DATASET")
    print("=" * 60)
    
    all_data = []
    
    # 1. Generate enhanced conversational dataset
    print("\n📊 Generating enhanced conversational data...")
    generate_enhanced_dataset(2000)
    with open("data/enhanced_corporate_training.jsonl", 'r') as f:
        enhanced_data = [json.loads(line) for line in f]
    all_data.extend(enhanced_data)
    print(f"✓ Added {len(enhanced_data)} conversational examples")
    
    # 2. Generate comprehensive 3-layer dataset
    print("\n📚 Generating comprehensive template data...")
    save_comprehensive_dataset(num_examples=2000)
    with open("data/comprehensive_corporate_training.jsonl", 'r') as f:
        comprehensive_data = [json.loads(line) for line in f]
    all_data.extend(comprehensive_data)
    print(f"✓ Added {len(comprehensive_data)} template-based examples")
    
    # 3. Generate decoder (reverse translation) dataset
    print("\n🔄 Generating decoder data...")
    generate_decoder_dataset(num_examples=2000)
    with open("data/decoder_training.jsonl", 'r') as f:
        decoder_data = [json.loads(line) for line in f]
    all_data.extend(decoder_data)
    print(f"✓ Added {len(decoder_data)} reverse translation examples")
    
    # 4. Generate combined dataset
    print("\n🎯 Generating combined hybrid data...")
    bot = CombinedCorporateBot()
    combined_data = bot.generate_combined_dataset(1000)
    all_data.extend(combined_data)
    print(f"✓ Added {len(combined_data)} hybrid examples")
    
    # 5. Add bidirectional examples (same content, both directions)
    print("\n↔️ Generating bidirectional pairs...")
    bidirectional_data = generate_bidirectional_examples(500)
    all_data.extend(bidirectional_data)
    print(f"✓ Added {len(bidirectional_data)} bidirectional examples")
    
    # Shuffle and save
    random.shuffle(all_data)
    
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ FINAL DATASET COMPLETE")
    print(f"📁 Total examples: {len(all_data)}")
    print(f"💾 Saved to: {output_path}")
    
    # Analyze dataset composition
    analyze_dataset(all_data)
    
    return output_path

def generate_bidirectional_examples(num_pairs: int = 500) -> list:
    """Generate examples that work in both directions"""
    
    bidirectional_pairs = [
        ("let's meet", "Let's sync up to align on our objectives"),
        ("need help", "I require assistance with these deliverables"),
        ("good job", "Excellent execution on those initiatives"),
        ("i'm busy", "My bandwidth is currently limited"),
        ("talk later", "Let's circle back on this"),
        ("send the file", "Could you share the deliverables?"),
        ("what's next?", "What are our next action items?"),
        ("any updates?", "Do you have visibility on the current status?"),
        ("i disagree", "I have a different perspective on this approach"),
        ("let's start", "Let's operationalize this initiative"),
        ("fix the problem", "We need to address these challenges"),
        ("make it better", "Let's optimize this for better outcomes"),
        ("work together", "Let's synergize our efforts"),
        ("tell everyone", "We should cascade this information"),
        ("think about it", "Let's ideate on potential solutions"),
    ]
    
    examples = []
    
    for _ in range(num_pairs // len(bidirectional_pairs)):
        for casual, corporate in bidirectional_pairs:
            # Forward direction
            examples.append({
                "instruction": "Transform to corporate speak",
                "input": casual,
                "output": corporate,
                "bidirectional": True
            })
            
            # Reverse direction
            examples.append({
                "instruction": "Translate corporate speak to plain English",
                "input": corporate,
                "output": casual,
                "bidirectional": True
            })
    
    return examples

def analyze_dataset(data: list):
    """Analyze the final dataset composition"""
    
    print("\n📊 DATASET ANALYSIS")
    print("=" * 60)
    
    # Instruction types
    instructions = {}
    for item in data:
        inst_type = item['instruction'].split()[0:3]
        inst_key = " ".join(inst_type)
        instructions[inst_key] = instructions.get(inst_key, 0) + 1
    
    print("\n📝 Instruction Types:")
    for inst, count in sorted(instructions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {inst}: {count}")
    
    # Context analysis
    contexts = {"domains": {}, "scenarios": {}, "seniority": {}}
    bidirectional_count = 0
    
    for item in data:
        if item.get('bidirectional'):
            bidirectional_count += 1
            
        if 'context' in item:
            if 'domain' in item['context']:
                d = item['context']['domain']
                contexts['domains'][d] = contexts['domains'].get(d, 0) + 1
            if 'scenario' in item['context']:
                s = item['context']['scenario']
                contexts['scenarios'][s] = contexts['scenarios'].get(s, 0) + 1
            if 'seniority' in item['context']:
                sn = item['context']['seniority']
                contexts['seniority'][sn] = contexts['seniority'].get(sn, 0) + 1
    
    print("\n🏢 Domains:")
    for domain, count in sorted(contexts['domains'].items()):
        print(f"  {domain}: {count}")
    
    print("\n🎯 Scenarios:")
    for scenario, count in sorted(contexts['scenarios'].items()):
        print(f"  {scenario}: {count}")
    
    print(f"\n↔️ Bidirectional examples: {bidirectional_count}")
    
    # Sample diversity
    print("\n🌟 Sample Diversity:")
    samples = random.sample(data, min(5, len(data)))
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. {sample['instruction']}")
        print(f"   In: {sample['input'][:60]}...")
        print(f"   Out: {sample['output'][:60]}...")

def create_training_configs():
    """Create configuration files for different training approaches"""
    
    # Config for LoRA fine-tuning
    lora_config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "dataset": "data/final_corporate_training.jsonl",
        "output_dir": "corporate-speak-lora-final",
        "training_args": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "eval_steps": 500
        },
        "lora_config": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }
    }
    
    # Config for full fine-tuning
    full_config = {
        "model_name": "microsoft/DialoGPT-large",
        "dataset": "data/final_corporate_training.jsonl",
        "output_dir": "corporate-speak-full",
        "training_args": {
            "num_epochs": 2,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "warmup_steps": 500
        }
    }
    
    # Save configs
    with open("config_lora.json", "w") as f:
        json.dump(lora_config, f, indent=2)
    
    with open("config_full.json", "w") as f:
        json.dump(full_config, f, indent=2)
    
    print("\n⚙️ Created training configuration files:")
    print("  - config_lora.json (for LoRA fine-tuning)")
    print("  - config_full.json (for full fine-tuning)")

if __name__ == "__main__":
    # Generate the final comprehensive dataset
    final_path = merge_all_datasets()
    
    # Create training configurations
    create_training_configs()
    
    print("\n🎉 DATASET GENERATION COMPLETE!")
    print("\n📋 Next Steps:")
    print("1. Upload data/final_corporate_training.jsonl to Hugging Face")
    print("2. Use config_lora.json for efficient LoRA training")
    print("3. Deploy to phxdev/corporate-speak-lora")
    print("\n💡 The model will handle:")
    print("  ✓ Casual → Corporate transformation")
    print("  ✓ Corporate → Plain English decoding")
    print("  ✓ Domain-specific language (6 industries)")
    print("  ✓ Seniority-aware communication (4 levels)")
    print("  ✓ Multi-turn conversations")
    print("  ✓ Specialized scenarios (interviews, roadmaps, etc.)")