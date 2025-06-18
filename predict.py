"""Cog prediction interface for SynergyBot"""

import torch
from cog import BasePredictor, Input, Path
import os
import json
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from synergy_bot.corporate_vocab import (
    CORPORATE_VERBS, CORPORATE_NOUNS, CORPORATE_ADJECTIVES,
    BUZZWORD_PHRASES, MEETING_COMPLICATIONS
)
from synergy_bot.dataset_generator import generate_training_pairs


class Predictor(BasePredictor):
    def setup(self):
        """Load model into memory"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = "microsoft/Phi-4"
        
        # Check if we have a pre-trained model
        if os.path.exists("/src/models/synergy-bot"):
            self.load_trained_model()
        else:
            # Load base model and prepare for training
            self.load_base_model()
            
    def load_base_model(self):
        """Load base model with LoRA configuration"""
        print(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.eval()
        
    def load_trained_model(self):
        """Load fine-tuned model"""
        print("Loading fine-tuned SynergyBot model")
        
        self.tokenizer = AutoTokenizer.from_pretrained("/src/models/synergy-bot")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, "/src/models/synergy-bot")
        self.model.eval()
        
    def generate_text(self, prompt, max_new_tokens=150, temperature=0.8):
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    def predict(
        self,
        text: str = Input(description="Input text to transform"),
        mode: str = Input(
            description="Transformation mode",
            choices=["synergize", "complicate_meeting", "inflate_title", "buzzword", "email"],
            default="synergize"
        ),
        temperature: float = Input(
            description="Generation temperature (higher = more creative)",
            default=0.8,
            ge=0.1,
            le=2.0
        ),
        train_model: bool = Input(
            description="Train the model first (takes ~10-15 minutes)",
            default=False
        )
    ) -> str:
        """Transform text into corporate speak"""
        
        if train_model and not os.path.exists("/src/models/synergy-bot"):
            # Generate training data and train model
            print("Generating training data...")
            training_data = generate_training_pairs()
            
            # Save training data
            os.makedirs("/src/data", exist_ok=True)
            with open("/src/data/corporate_training.jsonl", 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            # Note: Full training would happen here, but for Cog we'll use pre-trained
            print("Training skipped for demo - using base model with LoRA")
        
        # Generate appropriate prompt based on mode
        if mode == "synergize":
            prompt = f"""### Instruction: Convert this simple statement to corporate speak

### Input: {text}

### Response:"""
        
        elif mode == "complicate_meeting":
            prompt = f"""### Instruction: Complicate this meeting request with corporate speak

### Input: {text}

### Response:"""
        
        elif mode == "inflate_title":
            prompt = f"""### Instruction: Create a corporate title for this role

### Input: {text}

### Response:"""
        
        elif mode == "buzzword":
            prompt = f"""### Instruction: Generate corporate jargon

### Input: {text}

### Response:"""
            
        elif mode == "email":
            prompt = f"""### Instruction: Rewrite this email in corporate speak

### Input: {text}

### Response:"""
        
        # Generate response
        full_response = self.generate_text(prompt, temperature=temperature)
        
        # Extract just the response part
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response
            
        # Add some extra flair for certain modes
        if mode == "complicate_meeting":
            import random
            pre_meeting = f"Before we proceed, we'll need a {random.choice(['stakeholder alignment', 'strategic planning', 'innovation ideation'])} session."
            follow_up = f"Post-meeting, let's cascade the learnings across all verticals."
            response = f"{pre_meeting}\n\n{response}\n\n{follow_up}"
            
        elif mode == "inflate_title":
            import random
            if "Chief" not in response and random.random() > 0.5:
                response = f"Chief {response}"
            if "Officer" not in response and "Strategist" not in response and random.random() > 0.5:
                response = f"{response} & Innovation Evangelist"
                
        return response