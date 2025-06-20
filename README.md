# Corporate Synergy Bot 7B 🏢

A sophisticated language model that transforms casual communication into professional corporate speak and vice versa. Built with LoRA fine-tuning on Mistral-7B.

## Features 🚀

- **Bidirectional Translation**: Casual ↔️ Corporate speak
- **Domain Expertise**: Tech, Finance, Consulting, Healthcare, Retail, Manufacturing
- **Seniority Awareness**: Junior to Executive communication styles
- **Multi-turn Conversations**: Context-aware dialogue support

## Quick Start

### Use the Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "phxdev/corporate-synergy-bot-7b")

# Transform text
prompt = "Transform to corporate speak\nInput: let's meet\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Try the Demo

🌐 **Live Demo**: [Coming Soon - Space Link]

### Examples

**Casual → Corporate:**
- "let's meet" → "Let's sync up to align on our objectives"
- "good job" → "Excellent execution on those deliverables"
- "I'm busy" → "My bandwidth is currently limited"

**Corporate → Casual:**
- "Let's circle back on this" → "Let's talk about this later"
- "We need to leverage our synergies" → "We need to work together"

## Project Structure

```
synergy-bot/
├── src/                    # Core modules
│   ├── enhanced_corporate_bot.py
│   ├── corporate_decoder.py
│   └── combined_corporate_bot.py
├── training/               # Training scripts
│   ├── autotrain_setup.py
│   ├── train_lora.py
│   └── train_cpu.py
├── demo/                   # Demo applications
│   ├── app.py             # Gradio interface
│   └── demo_bidirectional.py
├── utils/                  # Utility scripts
│   └── generate_final_dataset.py
└── data/                   # Training data
```

## Training Your Own Model

### Dataset
- **Hugging Face Dataset**: [phxdev/corporate-speak-dataset](https://huggingface.co/datasets/phxdev/corporate-speak-dataset)
- 7,953 examples with bidirectional translations
- Domain and seniority annotations

### Training Options

1. **AutoTrain (Easiest)**
   ```bash
   python training/autotrain_setup.py
   # Follow the generated instructions
   ```

2. **Google Colab**
   - Use `training/train_colab.py`
   - Free T4 GPU available

3. **Local Training**
   ```bash
   pip install -r requirements.txt
   python training/train_lora.py
   ```

## Model Details

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Parameters**: r=16, alpha=32
- **Training Time**: ~2-3 hours on T4 GPU

## Citation

```bibtex
@misc{corporate-synergy-bot,
  author = {phxdev},
  title = {Corporate Synergy Bot 7B},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/phxdev/corporate-synergy-bot-7b}
}
```

## License

Apache 2.0

---

*Remember: To maximize stakeholder value, we must leverage our synergies through collaborative paradigm shifts! 😄*