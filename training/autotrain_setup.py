"""Setup script for Hugging Face AutoTrain"""

import json
from datetime import datetime

def create_autotrain_config():
    """Create configuration for AutoTrain"""
    
    config = {
        "task": "text-generation",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "dataset": "phxdev/corporate-speak-dataset",
        "column_mapping": {
            "text": "text"  # The column containing the formatted text
        },
        "project_name": f"corporate-synergy-bot-7b-{datetime.now().strftime('%Y%m%d')}",
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "warmup_ratio": 0.1,
        "gradient_accumulation": 4,
        "mixed_precision": "fp16",
        "push_to_hub": True,
        "hub_model_id": "phxdev/corporate-synergy-bot-7b",
        "hub_private": False,
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    with open("autotrain_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ AutoTrain configuration created: autotrain_config.json")
    return config

def generate_autotrain_cli_command(config):
    """Generate the CLI command for AutoTrain"""
    
    cmd = f"""autotrain llm \\
    --train \\
    --model {config['base_model']} \\
    --data-path phxdev/corporate-speak-dataset \\
    --text-column text \\
    --batch-size {config['batch_size']} \\
    --epochs {config['num_epochs']} \\
    --lr {config['learning_rate']} \\
    --warmup-ratio {config['warmup_ratio']} \\
    --gradient-accumulation {config['gradient_accumulation']} \\
    --mixed-precision {config['mixed_precision']} \\
    --use-peft \\
    --lora-r {config['lora_r']} \\
    --lora-alpha {config['lora_alpha']} \\
    --lora-dropout {config['lora_dropout']} \\
    --project-name {config['project_name']} \\
    --push-to-hub \\
    --hub-model-id {config['hub_model_id']}"""
    
    return cmd

def create_space_app():
    """Create app.py for Hugging Face Space"""
    
    app_content = '''import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_model = "phxdev/corporate-synergy-bot-7b"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_model)

def transform_text(text, mode="To Corporate", domain="general", seniority="mid"):
    """Transform text between casual and corporate speak"""
    
    if mode == "To Corporate":
        instruction = f"Transform to {domain} corporate speak (seniority: {seniority})"
    else:
        instruction = "Translate corporate speak to plain English"
    
    prompt = f"""### Instruction: {instruction}
### Input: {text}
### Response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Create Gradio interface
with gr.Blocks(title="Corporate Synergy Bot 7B") as demo:
    gr.Markdown("""
    # 🏢 Corporate Synergy Bot 7B
    
    Transform casual language into professional corporate communication or decode corporate jargon back to plain English.
    
    Powered by fine-tuned Mistral-7B with LoRA.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to transform...",
                lines=3
            )
            
            mode = gr.Radio(
                choices=["To Corporate", "To Plain English"],
                value="To Corporate",
                label="Transformation Mode"
            )
            
            with gr.Row():
                domain = gr.Dropdown(
                    choices=["general", "tech", "finance", "consulting", "healthcare", "retail"],
                    value="general",
                    label="Domain (for corporate mode)"
                )
                
                seniority = gr.Dropdown(
                    choices=["junior", "mid", "senior", "executive"],
                    value="mid",
                    label="Seniority Level"
                )
            
            transform_btn = gr.Button("Transform", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Transformed Text",
                lines=3
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["let's meet tomorrow", "To Corporate", "general", "mid"],
            ["I need help with this project", "To Corporate", "tech", "senior"],
            ["good job on the presentation", "To Corporate", "consulting", "executive"],
            ["We need to leverage our synergies to maximize stakeholder value", "To Plain English", "general", "mid"],
            ["Let's circle back on the deliverables", "To Plain English", "general", "mid"],
        ],
        inputs=[input_text, mode, domain, seniority]
    )
    
    transform_btn.click(
        fn=transform_text,
        inputs=[input_text, mode, domain, seniority],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
'''
    
    with open("app.py", "w") as f:
        f.write(app_content)
    
    print("✅ Gradio app created: app.py")

def create_requirements_txt():
    """Create requirements.txt for Space"""
    
    requirements = """gradio
transformers
torch
peft
accelerate
sentencepiece
protobuf
"""
    
    with open("requirements_space.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Requirements file created: requirements_space.txt")

def main():
    print("🚀 HUGGING FACE AUTOTRAIN SETUP")
    print("=" * 60)
    
    # Create configuration
    config = create_autotrain_config()
    
    # Generate CLI command
    cli_cmd = generate_autotrain_cli_command(config)
    
    print("\n📋 OPTION 1: AutoTrain Web UI")
    print("-" * 40)
    print("1. Go to: https://ui.autotrain.huggingface.co/")
    print("2. Create new project with these settings:")
    print(f"   - Task: Text Generation")
    print(f"   - Model: {config['base_model']}")
    print(f"   - Dataset: {config['dataset']}")
    print(f"   - Column: text")
    print(f"   - Use PEFT: Yes")
    print(f"   - LoRA r: {config['lora_r']}")
    print(f"   - LoRA alpha: {config['lora_alpha']}")
    
    print("\n📋 OPTION 2: AutoTrain CLI")
    print("-" * 40)
    print("Install AutoTrain:")
    print("pip install autotrain-advanced")
    print("\nRun this command:")
    print(cli_cmd)
    
    print("\n📋 OPTION 3: Google Colab")
    print("-" * 40)
    print("1. Create new Colab notebook")
    print("2. Select GPU runtime")
    print("3. Run these cells:")
    print("""
```python
# Cell 1: Install dependencies
!pip install autotrain-advanced transformers peft datasets accelerate

# Cell 2: Login to Hugging Face
from huggingface_hub import login
login()

# Cell 3: Run AutoTrain
!autotrain llm \\
    --train \\
    --model mistralai/Mistral-7B-Instruct-v0.2 \\
    --data-path phxdev/corporate-speak-dataset \\
    --text-column text \\
    --batch-size 4 \\
    --epochs 3 \\
    --lr 2e-4 \\
    --warmup-ratio 0.1 \\
    --gradient-accumulation 4 \\
    --mixed-precision fp16 \\
    --use-peft \\
    --lora-r 16 \\
    --lora-alpha 32 \\
    --lora-dropout 0.1 \\
    --project-name corporate-synergy-bot-7b \\
    --push-to-hub \\
    --hub-model-id phxdev/corporate-synergy-bot-7b
```
    """)
    
    # Create demo app
    create_space_app()
    create_requirements_txt()
    
    print("\n✅ All files prepared!")
    print("\n🎯 Next Steps:")
    print("1. Choose your training method (Web UI recommended for beginners)")
    print("2. Training will take ~2-3 hours on a T4 GPU")
    print("3. Model will be available at: https://huggingface.co/phxdev/corporate-synergy-bot-7b")
    print("\n💡 To create a demo Space after training:")
    print("1. Create new Space at: https://huggingface.co/new-space")
    print("2. Upload app.py and requirements_space.txt")
    print("3. Your bot will be live!")

if __name__ == "__main__":
    main()