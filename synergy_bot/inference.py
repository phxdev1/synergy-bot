"""Inference module for generating corporate speak with fine-tuned model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random

class SynergyBot:
    def __init__(self, model_path="models/synergy-bot", base_model="microsoft/Phi-4"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path, base_model)
        
    def load_model(self, model_path, base_model):
        """Load fine-tuned model with LoRA weights"""
        print(f"Loading SynergyBot from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print("SynergyBot loaded and ready to synergize!")
        
    def generate_corporate_speak(self, simple_text, instruction="Convert this simple statement to corporate speak"):
        """Transform simple text into corporate jargon"""
        prompt = f"""### Instruction: {instruction}

### Input: {simple_text}

### Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part
        response = response.split("### Response:")[-1].strip()
        
        return response
    
    def complicate_meeting(self, meeting_request):
        """Turn simple meeting request into corporate nightmare"""
        return self.generate_corporate_speak(
            meeting_request,
            instruction="Complicate this meeting request with corporate speak"
        )
    
    def generate_corporate_title(self, base_role):
        """Generate inflated corporate title"""
        return self.generate_corporate_speak(
            base_role,
            instruction="Create a corporate title for this role"
        )
    
    def buzzword_generator(self, topic="business strategy"):
        """Generate pure corporate buzzword salad"""
        return self.generate_corporate_speak(
            topic,
            instruction="Generate corporate jargon"
        )
    
    def synergize_email(self, email_content):
        """Transform normal email into corporate speak"""
        lines = email_content.strip().split('\n')
        synergized_lines = []
        
        for line in lines:
            if line.strip():
                synergized = self.generate_corporate_speak(line)
                synergized_lines.append(synergized)
            else:
                synergized_lines.append("")
                
        return '\n'.join(synergized_lines)

class MeetingComplicator:
    """Specialized module for making meetings unnecessarily complex"""
    
    def __init__(self, bot):
        self.bot = bot
        self.complications = [
            "stakeholder alignment session",
            "cross-functional sync",
            "strategic planning workshop",
            "innovation ideation summit",
            "transformation steering committee"
        ]
        
    def complicate(self, original_meeting):
        """Turn simple meeting into multi-step process"""
        base = self.bot.complicate_meeting(original_meeting)
        
        # Add pre-meetings
        pre_meeting = f"Before we can {original_meeting}, we'll need a {random.choice(self.complications)} to align on objectives."
        
        # Add follow-ups
        follow_up = f"Post-meeting, we'll schedule a debrief to cascade learnings and socialize next steps with key stakeholders."
        
        return f"{pre_meeting}\n\n{base}\n\n{follow_up}"

class TitleInflator:
    """Generate increasingly ridiculous corporate titles"""
    
    def __init__(self, bot):
        self.bot = bot
        self.prefixes = ["Chief", "Senior", "Global", "Executive", "Principal"]
        self.suffixes = ["Strategist", "Evangelist", "Architect", "Officer", "Lead"]
        
    def inflate(self, base_title):
        """Create multi-level inflated title"""
        # First pass through model
        inflated = self.bot.generate_corporate_title(base_title)
        
        # Add extra fluff
        prefix = random.choice(self.prefixes)
        suffix = random.choice(self.suffixes)
        
        if prefix not in inflated:
            inflated = f"{prefix} {inflated}"
        if suffix not in inflated:
            inflated = f"{inflated} & {suffix}"
            
        return inflated