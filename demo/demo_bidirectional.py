"""Bidirectional Corporate Speak Demo - Forward and Reverse Translation"""

from corporate_decoder import CorporateDecoder
from enhanced_corporate_bot import ConversationContext, Domain, SeniorityLevel
import random

class BidirectionalCorporateBot:
    """Demo bot showing both corporate encoding and decoding"""
    
    def __init__(self):
        self.decoder = CorporateDecoder()
        
        # Simple forward translations for demo
        self.forward_mappings = {
            "let's meet": "Let's sync up to align on our objectives",
            "need help": "I require assistance with these deliverables",
            "good job": "Excellent execution on those initiatives",
            "i'm busy": "My bandwidth is currently limited",
            "talk later": "Let's circle back on this",
            "send the file": "Could you share the deliverables?",
            "what's next": "What are our next action items?",
            "any updates": "Do you have visibility on the current status?",
            "i disagree": "I have a different perspective on this approach",
            "let's start": "Let's operationalize this initiative",
        }
    
    def corporate_encode(self, casual_text: str, domain: str = None) -> str:
        """Transform casual to corporate speak"""
        # Check for exact matches first
        casual_lower = casual_text.lower().strip()
        if casual_lower in self.forward_mappings:
            return self.forward_mappings[casual_lower]
        
        # Otherwise, apply transformations
        corporate = casual_text
        
        # Simple word replacements
        replacements = {
            "help": "assistance",
            "talk": "discuss",
            "meet": "sync up",
            "think": "believe",
            "fix": "address",
            "problem": "challenge",
            "work": "deliverables",
            "do": "execute",
            "start": "initiate",
            "end": "conclude",
            "get": "obtain",
            "send": "share",
            "tell": "inform",
            "ask": "inquire",
        }
        
        for casual, corp in replacements.items():
            corporate = corporate.replace(casual, corp)
        
        # Add domain flavor if specified
        if domain:
            domain_phrases = {
                "tech": " leveraging our agile methodology",
                "finance": " ensuring regulatory compliance",
                "consulting": " aligned with best practices",
                "healthcare": " maintaining patient-centered focus",
            }
            if domain in domain_phrases and len(corporate) < 50:
                corporate += domain_phrases[domain]
        
        return corporate
    
    def demonstrate_bidirectional(self):
        """Show examples of bidirectional translation"""
        print("🔄 BIDIRECTIONAL CORPORATE SPEAK TRANSLATOR")
        print("=" * 60)
        
        examples = [
            ("let's meet tomorrow", None),
            ("i need help with this project", "tech"),
            ("good job on the presentation", None),
            ("we have a problem", "finance"),
            ("what's the status?", "consulting"),
        ]
        
        for casual, domain in examples:
            # Forward translation
            corporate = self.corporate_encode(casual, domain)
            
            # Reverse translation
            decoded = self.decoder.decode(corporate)
            
            print(f"\n📝 Original: {casual}")
            if domain:
                print(f"   Domain: {domain}")
            print(f"🏢 Corporate: {corporate}")
            print(f"💬 Decoded: {decoded}")
            print(f"✓ Round-trip similarity: {self._similarity(casual, decoded)}%")
    
    def _similarity(self, text1: str, text2: str) -> int:
        """Calculate simple similarity percentage"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1:
            return 0
        common = words1.intersection(words2)
        return int((len(common) / len(words1)) * 100)
    
    def interactive_translator(self):
        """Interactive translation demo"""
        print("\n\n💼 INTERACTIVE CORPORATE TRANSLATOR")
        print("=" * 60)
        print("Commands:")
        print("  corp: <text>  - Translate to corporate speak")
        print("  plain: <text> - Translate to plain English")
        print("  demo          - Show more examples")
        print("  quit          - Exit")
        print()
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! Remember to leverage your synergies! 👋")
                break
            
            elif user_input.lower() == 'demo':
                self.demonstrate_bidirectional()
            
            elif user_input.lower().startswith('corp:'):
                text = user_input[5:].strip()
                corporate = self.corporate_encode(text)
                print(f"🏢 Corporate: {corporate}")
            
            elif user_input.lower().startswith('plain:'):
                text = user_input[6:].strip()
                plain = self.decoder.decode(text)
                print(f"💬 Plain: {plain}")
            
            else:
                # Default to corporate translation
                corporate = self.corporate_encode(user_input)
                print(f"🏢 Corporate: {corporate}")
                
                # Also show reverse
                decoded = self.decoder.decode(corporate)
                print(f"💬 Decoded back: {decoded}")

def test_edge_cases():
    """Test edge cases and complex translations"""
    bot = BidirectionalCorporateBot()
    decoder = bot.decoder
    
    print("\n\n🧪 TESTING EDGE CASES")
    print("=" * 60)
    
    test_cases = [
        # Multi-layer corporate speak
        "We need to leverage our bandwidth to synergize the deliverables",
        
        # Question forms
        "Could you provide visibility into the project's action items?",
        
        # Time-sensitive
        "Let's circle back on this by EOD to ensure stakeholder alignment",
        
        # Negatives
        "I don't have the cycles to deep dive into this initiative",
        
        # Complex sentence
        "Going forward, we should operationalize our learnings to optimize the customer journey",
    ]
    
    for corp_text in test_cases:
        plain = decoder.decode(corp_text)
        print(f"\n🏢 Corporate: {corp_text}")
        print(f"💬 Plain: {plain}")
        
        # Try to re-encode
        re_encoded = bot.corporate_encode(plain)
        print(f"🔄 Re-encoded: {re_encoded}")

def generate_conversation_pairs():
    """Generate conversational exchange examples"""
    print("\n\n💬 CONVERSATION TRANSLATION EXAMPLES")
    print("=" * 60)
    
    conversations = [
        {
            "context": "Project Update Meeting",
            "exchanges": [
                ("Hey, how's the project going?", "I wanted to touch base on the project status."),
                ("We hit some problems but we're fixing them", "We've encountered some challenges but we're actively addressing them."),
                ("When will it be done?", "What's the anticipated timeline for completion?"),
                ("Maybe next week if all goes well", "We're targeting next week, pending successful resolution of current blockers."),
            ]
        },
        {
            "context": "Performance Review",
            "exchanges": [
                ("You did great this quarter", "Your performance this quarter exceeded expectations."),
                ("Thanks, I worked hard", "I appreciate the feedback. I've been focused on delivering strong results."),
                ("Keep it up", "Continue leveraging your strengths to drive value."),
                ("Will do!", "I'm committed to maintaining this trajectory."),
            ]
        }
    ]
    
    decoder = CorporateDecoder()
    
    for conv in conversations:
        print(f"\n📋 Context: {conv['context']}")
        print("-" * 40)
        
        for i, (casual, corporate) in enumerate(conv['exchanges'], 1):
            decoded = decoder.decode(corporate)
            
            print(f"\nExchange {i}:")
            print(f"  😊 Casual: {casual}")
            print(f"  🏢 Corporate: {corporate}")
            print(f"  🔄 Decoded: {decoded}")

if __name__ == "__main__":
    # Create bot instance
    bot = BidirectionalCorporateBot()
    
    # Run demonstrations
    bot.demonstrate_bidirectional()
    test_edge_cases()
    generate_conversation_pairs()
    
    # Start interactive mode
    bot.interactive_translator()