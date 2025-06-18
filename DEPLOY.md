# Deploying SynergyBot to Replicate

## Prerequisites

1. Create a Replicate account at https://replicate.com
2. Get your API token from https://replicate.com/account/api-tokens

## Deployment Steps

1. **Authenticate with Replicate:**
```bash
./cog login
```

2. **Build the container (if not already done):**
```bash
./cog build
```

3. **Push to Replicate:**
```bash
./cog push r8.im/phxdev1/synergy-bot
```

## Using the Deployed Model

Once deployed, you can use the model via:

### Web Interface
Visit: https://replicate.com/phxdev1/synergy-bot

### API
```python
import replicate

output = replicate.run(
    "phxdev1/synergy-bot:latest",
    input={
        "text": "let's have a meeting",
        "mode": "complicate_meeting",
        "temperature": 0.8
    }
)
print(output)
```

### CLI
```bash
./cog predict r8.im/phxdev1/synergy-bot:latest \
  -i text="we need to improve sales" \
  -i mode="synergize"
```

## Available Modes

- `synergize`: Convert simple text to corporate speak
- `complicate_meeting`: Turn meetings into multi-stakeholder nightmares
- `inflate_title`: Create ridiculous corporate titles
- `buzzword`: Generate pure buzzword salad
- `email`: Transform emails into corporate speak

## Parameters

- `text`: Input text to transform (required)
- `mode`: Transformation mode (default: "synergize")
- `temperature`: Generation creativity (0.1-2.0, default: 0.8)
- `train_model`: Train model first (default: false)

## Example Outputs

**Input:** "let's fix this bug"
**Output:** "Let's leverage our agile methodologies to optimize system functionality while ensuring stakeholder alignment"

**Input:** "developer" (mode: inflate_title)
**Output:** "Chief Digital Transformation Architect & Innovation Evangelist"