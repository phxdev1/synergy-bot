"""Corporate Speak Decoder - Comprehensive reverse translation"""

import re
import json
import random
from typing import Dict, List, Tuple

class CorporateDecoder:
    """Decode corporate speak back to plain English"""
    
    def __init__(self):
        # Comprehensive corporate-to-plain mappings
        self.phrase_mappings = {
            # Meeting and communication phrases
            "let's circle back": ["let's talk later", "we'll discuss this later"],
            "touch base": ["talk", "check in", "connect"],
            "reach out": ["contact", "call", "email"],
            "loop in": ["include", "add", "tell"],
            "take this offline": ["discuss privately", "talk later"],
            "sync up": ["meet", "talk", "coordinate"],
            "align on": ["agree on", "decide on"],
            "ping": ["message", "contact", "remind"],
            "cascade": ["share", "tell everyone", "pass down"],
            "socialize": ["share", "discuss", "spread"],
            
            # Capability and capacity
            "bandwidth": ["time", "capacity", "availability"],
            "cycles": ["time", "effort", "resources"],
            "leverage": ["use", "utilize", "take advantage of"],
            "utilize": ["use"],
            "optimize": ["improve", "make better", "enhance"],
            "maximize": ["increase", "get the most"],
            "synergize": ["work together", "collaborate", "combine"],
            
            # Business jargon
            "deliverables": ["work", "results", "outputs"],
            "action items": ["tasks", "to-dos", "next steps"],
            "action item": ["task", "to-do", "next step"],
            "low-hanging fruit": ["easy tasks", "quick wins", "simple stuff"],
            "move the needle": ["make progress", "have impact", "make a difference"],
            "boil the ocean": ["do too much", "overcomplicate", "waste effort"],
            "drill down": ["look deeper", "get details", "examine closely"],
            "deep dive": ["detailed review", "thorough analysis", "close look"],
            "30,000 foot view": ["overview", "big picture", "summary"],
            
            # Time and scheduling
            "by eob": ["by end of day", "today"],
            "by eod": ["by end of day", "today"],
            "at your earliest convenience": ["when you can", "soon"],
            "going forward": ["from now on", "in the future"],
            "on my radar": ["I'm aware", "I know about it"],
            "in the pipeline": ["coming soon", "being worked on"],
            
            # Corporate actions
            "operationalize": ["put into practice", "implement", "start doing"],
            "incentivize": ["encourage", "motivate", "reward"],
            "right-size": ["adjust", "resize", "cut"],
            "sunset": ["end", "discontinue", "stop"],
            "pivot": ["change direction", "switch", "try something else"],
            "iterate": ["improve", "refine", "try again"],
            
            # Agreement and understanding
            "aligned": ["in agreement", "on the same page"],
            "on the same page": ["in agreement", "understand each other"],
            "buy-in": ["agreement", "support", "approval"],
            "stakeholder alignment": ["everyone agrees", "consensus"],
            
            # Business outcomes
            "value add": ["benefit", "improvement", "help"],
            "roi": ["return", "benefit", "payoff"],
            "kpis": ["goals", "metrics", "targets"],
            "best practices": ["good ways", "standard methods", "what works"],
            "core competencies": ["what we're good at", "strengths"],
            "competitive advantage": ["edge", "what makes us better"],
            
            # Soft language
            "challenges": ["problems", "issues", "difficulties"],
            "opportunities": ["chances", "possibilities"],
            "learnings": ["lessons", "what we learned"],
            "asks": ["requests", "needs", "requirements"],
        }
        
        # Verb transformations
        self.verb_transformations = {
            "facilitate": "help with",
            "coordinate": "organize",
            "spearhead": "lead",
            "champion": "support",
            "evangelize": "promote",
            "streamline": "simplify",
            "enhance": "improve",
            "augment": "add to",
            "implement": "do",
            "execute": "do",
            "actualize": "make happen",
            "ideate": "think of ideas",
            "strategize": "plan",
            "conceptualize": "think about",
            "crystallize": "clarify",
        }
        
        # Contextual patterns (regex)
        self.pattern_replacements = [
            (r"leverage our (\w+) to", r"use our \1 to"),
            (r"optimize (\w+) for maximum", r"improve \1 for best"),
            (r"align our (\w+) with", r"match our \1 with"),
            (r"synergize (\w+) across", r"combine \1 across"),
            (r"operationalize the (\w+)", r"start using the \1"),
            (r"drive (\w+) forward", r"push \1 ahead"),
            (r"surface (\w+) insights", r"find \1 information"),
            (r"unlock (\w+) potential", r"use \1 better"),
            (r"activate (\w+) capabilities", r"start using \1 features"),
        ]
        
        # Sentence structure simplifications
        self.structural_patterns = [
            # Remove corporate fluff
            (r"at the end of the day,?\s*", ""),
            (r"going forward,?\s*", ""),
            (r"in order to\s+", "to "),
            (r"in regards to\s+", "about "),
            (r"with regards to\s+", "about "),
            (r"in terms of\s+", "for "),
            (r"from a (\w+) perspective", r"for \1"),
            (r"on a (\w+) basis", r"\1ly"),  # "on a daily basis" -> "daily"
            
            # Simplify corporate phrases
            (r"i wanted to touch base", "i wanted to talk"),
            (r"i'd like to circle back", "let's talk again"),
            (r"let me loop you in", "let me tell you"),
            (r"can we take this offline", "can we talk privately"),
            (r"i don't have the bandwidth", "i don't have time"),
            (r"let's sync up", "let's meet"),
            
            # Question simplifications
            (r"what are your thoughts on", "what do you think about"),
            (r"do you have any concerns regarding", "are you worried about"),
            (r"could you provide insight into", "can you explain"),
            (r"would you be able to", "can you"),
        ]
    
    def decode(self, corporate_text: str) -> str:
        """Decode corporate speak to plain English"""
        # Start with lowercase for easier matching
        decoded = corporate_text.lower()
        
        # Apply phrase mappings (longest phrases first to avoid partial matches)
        sorted_phrases = sorted(self.phrase_mappings.keys(), key=len, reverse=True)
        for corp_phrase in sorted_phrases:
            if corp_phrase in decoded:
                replacement = self.phrase_mappings[corp_phrase][0]  # Use first option
                decoded = decoded.replace(corp_phrase, replacement)
        
        # Apply verb transformations
        for corp_verb, simple_verb in self.verb_transformations.items():
            # Match verb forms (base, ing, ed, s)
            patterns = [
                (f"\\b{corp_verb}\\b", simple_verb),
                (f"\\b{corp_verb}s\\b", f"{simple_verb}s"),
                (f"\\b{corp_verb}d\\b", f"{simple_verb}d"),
                (f"\\b{corp_verb[:-1]}ing\\b", f"{simple_verb[:-1]}ing" if simple_verb.endswith('e') else f"{simple_verb}ing"),
            ]
            for pattern, replacement in patterns:
                decoded = re.sub(pattern, replacement, decoded)
        
        # Apply contextual patterns
        for pattern, replacement in self.pattern_replacements:
            decoded = re.sub(pattern, replacement, decoded)
        
        # Apply structural simplifications
        for pattern, replacement in self.structural_patterns:
            decoded = re.sub(pattern, replacement, decoded, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        decoded = re.sub(r'\s+', ' ', decoded).strip()
        
        # Restore sentence case
        if decoded:
            decoded = decoded[0].upper() + decoded[1:]
        
        return decoded
    
    def decode_conversation(self, messages: List[str]) -> List[Tuple[str, str]]:
        """Decode a series of corporate messages"""
        results = []
        for message in messages:
            decoded = self.decode(message)
            results.append((message, decoded))
        return results
    
    def generate_training_pairs(self, num_examples: int = 1000) -> List[Dict]:
        """Generate corporate->plain training pairs"""
        training_data = []
        
        # Template for generating corporate sentences
        corporate_templates = [
            "Let's {corp_verb} to {corp_action} our {corp_noun}",
            "We need to {corp_verb} the {corp_noun} to {corp_outcome}",
            "Can we {corp_action} on the {corp_noun} {time_phrase}?",
            "I'll {corp_action} with {stakeholder} about {corp_noun}",
            "We should {corp_verb} our {corp_noun} to ensure {corp_outcome}",
            "{time_phrase}, let's {corp_action} to discuss {corp_noun}",
            "I don't have the {capacity} to {corp_verb} the {corp_noun} right now",
            "We're {corp_progress} on {corp_verb}ing our {corp_noun}",
        ]
        
        # Corporate vocabulary for templates
        corp_verbs = ["leverage", "optimize", "synergize", "facilitate", "operationalize", "strategize"]
        corp_actions = ["touch base", "circle back", "sync up", "deep dive", "drill down", "reach out"]
        corp_nouns = ["deliverables", "action items", "initiatives", "objectives", "stakeholders", "outcomes"]
        corp_outcomes = ["maximize ROI", "drive value", "ensure alignment", "optimize performance", "achieve synergies"]
        time_phrases = ["going forward", "at your earliest convenience", "by EOD", "early next week"]
        capacity_terms = ["bandwidth", "cycles", "capacity", "resources"]
        progress_terms = ["making progress", "moving the needle", "gaining traction", "building momentum"]
        stakeholders = ["the team", "leadership", "stakeholders", "the client", "management"]
        
        # Generate examples
        for _ in range(num_examples):
            template = random.choice(corporate_templates)
            
            # Fill template with corporate speak
            corporate = template.format(
                corp_verb=random.choice(corp_verbs),
                corp_action=random.choice(corp_actions),
                corp_noun=random.choice(corp_nouns),
                corp_outcome=random.choice(corp_outcomes),
                time_phrase=random.choice(time_phrases),
                capacity=random.choice(capacity_terms),
                corp_progress=random.choice(progress_terms),
                stakeholder=random.choice(stakeholders)
            )
            
            # Decode to plain English
            plain = self.decode(corporate)
            
            training_data.append({
                "instruction": "Translate this corporate speak to plain English",
                "input": corporate,
                "output": plain
            })
        
        # Add specific examples for common patterns
        specific_examples = [
            ("Let me loop you in on our latest initiatives", "Let me tell you about our latest projects"),
            ("We need to leverage our core competencies", "We need to use what we're good at"),
            ("Can we take this offline?", "Can we talk privately?"),
            ("I don't have the bandwidth right now", "I don't have time right now"),
            ("Let's circle back on this next week", "Let's talk about this again next week"),
            ("We should cascade this information to all stakeholders", "We should share this information with everyone"),
            ("The team is aligned on the deliverables", "The team agrees on the work"),
            ("We're looking to optimize our processes going forward", "We're looking to improve our processes"),
            ("This will help us move the needle on our KPIs", "This will help us make progress on our goals"),
            ("Let's ideate on some solutions", "Let's think of some solutions"),
            ("We need to operationalize this strategy", "We need to start using this strategy"),
            ("Can you provide visibility into the project status?", "Can you update me on the project?"),
            ("We're pivoting our approach to maximize value", "We're changing our approach to get better results"),
            ("Let's sync up to align on next steps", "Let's meet to agree on what to do next"),
            ("I'll reach out to socialize this idea", "I'll contact people to share this idea"),
        ]
        
        for corp, plain in specific_examples:
            training_data.append({
                "instruction": "Translate this corporate speak to plain English",
                "input": corp,
                "output": plain
            })
        
        return training_data

def test_decoder():
    """Test the corporate decoder with examples"""
    decoder = CorporateDecoder()
    
    test_phrases = [
        "Let's circle back on this next week to ensure we're aligned on the deliverables.",
        "I don't have the bandwidth to deep dive into this right now.",
        "We need to leverage our core competencies to maximize ROI going forward.",
        "Can we take this offline? I'd like to sync up on our strategy.",
        "Let me loop in the stakeholders so we can operationalize this initiative.",
        "We're looking to optimize our processes to move the needle on our KPIs.",
        "At the end of the day, we need to ensure stakeholder buy-in.",
        "I'll reach out to the team to cascade this information.",
        "We should ideate on how to pivot our approach.",
        "Let's touch base early next week to drill down on the action items.",
    ]
    
    print("🔄 CORPORATE SPEAK DECODER TEST")
    print("=" * 60)
    
    for phrase in test_phrases:
        decoded = decoder.decode(phrase)
        print(f"\n📝 Corporate: {phrase}")
        print(f"💬 Plain: {decoded}")
    
    # Test conversation decoding
    print("\n\n📞 CONVERSATION DECODE TEST")
    print("=" * 60)
    
    corporate_conversation = [
        "Hi team, I wanted to touch base on our Q3 deliverables.",
        "We need to leverage our bandwidth to maximize impact.",
        "Can we sync up tomorrow to align on the action items?",
        "I'll loop in stakeholders to ensure we're all on the same page.",
    ]
    
    decoded_conversation = decoder.decode_conversation(corporate_conversation)
    
    for i, (corp, plain) in enumerate(decoded_conversation, 1):
        print(f"\nTurn {i}:")
        print(f"  Corporate: {corp}")
        print(f"  Plain: {plain}")

def generate_decoder_dataset(output_path: str = "data/decoder_training.jsonl", num_examples: int = 2000):
    """Generate decoder training dataset"""
    decoder = CorporateDecoder()
    training_data = decoder.generate_training_pairs(num_examples)
    
    # Save dataset
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Generated {len(training_data)} decoder training examples")
    print(f"📁 Saved to: {output_path}")
    
    # Show samples
    print("\n📝 Sample decoder training pairs:")
    for i, example in enumerate(random.sample(training_data, 5), 1):
        print(f"\n{i}. {example['instruction']}")
        print(f"   Input: {example['input']}")
        print(f"   Output: {example['output']}")

if __name__ == "__main__":
    # Test the decoder
    test_decoder()
    
    # Generate decoder training data
    generate_decoder_dataset()