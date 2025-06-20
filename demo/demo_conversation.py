"""Interactive demo for corporate speak conversation bot"""

import json
from enhanced_corporate_bot import (
    ConversationContext, Domain, SeniorityLevel, 
    ConversationGenerator, ReverseTranslator
)
# from train_lora import CorporateSpeakInference  # Uncomment when dependencies are installed

class InteractiveCorporateBot:
    def __init__(self, model_path: str = "corporate-speak-lora"):
        self.conversation_gen = ConversationGenerator()
        self.reverse_translator = ReverseTranslator()
        # Uncomment when model is trained
        # self.inference = CorporateSpeakInference(model_path)
        
        # Current conversation context
        self.context = ConversationContext(
            domain=Domain.TECH_STARTUP,
            seniority=SeniorityLevel.MID,
            relationship="peer",
            urgency=3
        )
        
        self.conversation_history = []
    
    def set_context(self, domain: str, seniority: str, relationship: str = "peer"):
        """Update conversation context"""
        domain_map = {
            "tech": Domain.TECH_STARTUP,
            "consulting": Domain.CONSULTING,
            "finance": Domain.FINANCE,
            "healthcare": Domain.HEALTHCARE,
        }
        
        seniority_map = {
            "junior": SeniorityLevel.JUNIOR,
            "mid": SeniorityLevel.MID,
            "senior": SeniorityLevel.SENIOR,
            "executive": SeniorityLevel.EXECUTIVE,
        }
        
        self.context = ConversationContext(
            domain=domain_map.get(domain, Domain.TECH_STARTUP),
            seniority=seniority_map.get(seniority, SeniorityLevel.MID),
            relationship=relationship,
            urgency=3
        )
        
        print(f"Context updated: {self.context.domain.value} industry, {self.context.seniority.name} level, {relationship} relationship")
    
    def translate_message(self, message: str, to_corporate: bool = True) -> str:
        """Translate a single message"""
        # In production, this would use the trained model
        # return self.inference.generate_corporate(message, self.context.domain.value, self.context.seniority.value)
        
        # For demo, use template-based generation
        if to_corporate:
            return self._demo_corporate_transform(message)
        else:
            return self._demo_casual_transform(message)
    
    def _demo_corporate_transform(self, casual: str) -> str:
        """Demo transformation without trained model"""
        transformations = {
            "hey": "Good morning/afternoon",
            "thanks": "I appreciate your assistance",
            "ok": "Understood, I'll proceed accordingly",
            "sure": "Certainly, I'd be happy to",
            "what's up": "I hope this message finds you well",
            "got it": "I acknowledge and understand",
            "my bad": "I apologize for the oversight",
            "asap": "at your earliest convenience",
            "fyi": "for your awareness",
            "btw": "additionally, I wanted to mention"
        }
        
        result = casual
        for casual_term, corporate_term in transformations.items():
            result = result.replace(casual_term, corporate_term)
        
        # Add domain-specific touches
        if self.context.domain == Domain.TECH_STARTUP:
            result = result.replace("project", "sprint")
            result = result.replace("problem", "blocker")
        elif self.context.domain == Domain.CONSULTING:
            result = result.replace("idea", "recommendation")
            result = result.replace("work", "deliverable")
        
        return result
    
    def _demo_casual_transform(self, corporate: str) -> str:
        """Demo reverse transformation"""
        for pattern, translations in self.reverse_translator.reverse_patterns.items():
            if pattern in corporate.lower():
                return corporate.lower().replace(pattern, translations[0])
        return corporate
    
    def simulate_conversation(self, conversation_type: str = "project_update"):
        """Simulate a full conversation"""
        conversation = self.conversation_gen.generate_conversation(
            self.context, conversation_type
        )
        
        print(f"\n--- Simulated {conversation_type.replace('_', ' ').title()} Conversation ---")
        print(f"Context: {self.context.domain.value}, {self.context.seniority.name} level\n")
        
        for i, turn in enumerate(conversation):
            print(f"Turn {i+1}:")
            print(f"  Casual: {turn['casual']}")
            print(f"  Corporate: {turn['corporate']}")
            
            if 'response' in turn:
                print(f"  Response (Casual): {turn['response']['casual']}")
                print(f"  Response (Corporate): {turn['response']['corporate']}")
            print()
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n=== Corporate Speak Conversation Bot ===")
        print("Commands:")
        print("  /context <domain> <seniority> - Set context (e.g., /context tech senior)")
        print("  /translate <message> - Translate to corporate speak")
        print("  /casual <message> - Translate to casual speak")
        print("  /simulate <type> - Simulate conversation (project_update, meeting_request, feedback_delivery)")
        print("  /examples - Show example transformations")
        print("  /quit - Exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.startswith("/quit"):
                print("Goodbye! Remember to leverage our synergies moving forward! 😄")
                break
            
            elif user_input.startswith("/context"):
                parts = user_input.split()
                if len(parts) >= 3:
                    self.set_context(parts[1], parts[2], parts[3] if len(parts) > 3 else "peer")
                else:
                    print("Usage: /context <domain> <seniority> [relationship]")
            
            elif user_input.startswith("/translate"):
                message = user_input[10:].strip()
                corporate = self.translate_message(message, to_corporate=True)
                print(f"Corporate: {corporate}")
            
            elif user_input.startswith("/casual"):
                message = user_input[7:].strip()
                casual = self.translate_message(message, to_corporate=False)
                print(f"Casual: {casual}")
            
            elif user_input.startswith("/simulate"):
                conv_type = user_input[9:].strip() or "project_update"
                self.simulate_conversation(conv_type)
            
            elif user_input.startswith("/examples"):
                self.show_examples()
            
            else:
                # Default: translate to corporate
                corporate = self.translate_message(user_input, to_corporate=True)
                print(f"Corporate: {corporate}")
    
    def show_examples(self):
        """Show example transformations for current context"""
        print(f"\n--- Example Transformations ({self.context.domain.value}, {self.context.seniority.name}) ---\n")
        
        examples = [
            ("hey, got time to chat?", "Do you have availability for a brief discussion?"),
            ("the project is behind", "The deliverables timeline has shifted"),
            ("I disagree", "I have a different perspective on this approach"),
            ("good job", "Excellent work on the implementation"),
            ("I'm swamped", "My bandwidth is currently limited"),
            ("let's fix this", "Let's collaborate to address these challenges"),
        ]
        
        for casual, corporate in examples:
            print(f"Casual: {casual}")
            print(f"Corporate: {corporate}\n")

def main():
    """Run the demo"""
    bot = InteractiveCorporateBot()
    
    # Show some examples first
    print("=== Corporate Speak Bot Demo ===\n")
    
    # Demo different contexts
    contexts = [
        ("tech", "junior", "superior"),
        ("consulting", "senior", "client"),
        ("finance", "executive", "peer"),
    ]
    
    for domain, seniority, relationship in contexts:
        bot.set_context(domain, seniority, relationship)
        bot.simulate_conversation("project_update")
        print("-" * 50)
    
    # Start interactive mode
    bot.interactive_mode()

if __name__ == "__main__":
    main()