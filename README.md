# SynergyBot 🏢

An LLM fine-tuned to generate corporate jargon, complicate meetings, and create meaningless titles. Transform simple communication into a corporate hellscape!

## Features

- **Corporate Speak Translation**: Convert simple statements into buzzword-laden corporate jargon
- **Meeting Complicator**: Turn simple meeting requests into multi-stakeholder alignment marathons  
- **Title Inflator**: Generate ridiculously inflated corporate titles
- **Buzzword Generator**: Create pure corporate buzzword salad on any topic

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage

### Train the Model

First, train the SynergyBot model on corporate speak:

```bash
synergy-bot train
```

This will:
1. Generate a training dataset of corporate jargon examples
2. Fine-tune a base LLM (default: Microsoft Phi-4) using LoRA
3. Save the model to `models/synergy-bot`

### Command Line Interface

Transform simple text:
```bash
synergy-bot synergize "let's fix this bug"
# Output: "Let's leverage our agile methodologies to optimize system functionality"
```

Complicate meetings:
```bash
synergy-bot complicate "can we meet tomorrow?"
# Output: Multiple paragraphs about stakeholder alignment, pre-meetings, and cascading action items
```

Generate corporate titles:
```bash
synergy-bot inflate "developer"
# Output: "Chief Digital Transformation Architect & Innovation Evangelist"
```

Create buzzword salad:
```bash
synergy-bot buzzwords --topic "product launch" --count 3
```

### Interactive Mode

```bash
synergy-bot interactive
```

## Architecture

The bot uses:
- **Base Model**: Microsoft Phi-4 (or any Hugging Face causal LM)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
- **Dataset**: Generated pairs of simple->corporate translations
- **Framework**: Transformers + PEFT

## Examples

**Simple Statement → Corporate Speak:**
- "good idea" → "paradigm-shifting value proposition"
- "I disagree" → "I think we need to pivot our approach to maximize synergies"
- "it's broken" → "we're experiencing suboptimal performance metrics"

**Meeting Complications:**
- "let's discuss" → Full stakeholder alignment process with pre-meetings and follow-ups

**Title Inflation:**
- "analyst" → "Senior Strategic Analytics Transformation Officer"

## Contributing

Feel free to add more corporate buzzwords, meeting complications, or training examples!