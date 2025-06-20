import gradio as gr
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
