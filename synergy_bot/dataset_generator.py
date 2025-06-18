"""Generate training dataset for corporate speak LLM"""

import json
import random
from itertools import combinations
from synergy_bot.corporate_vocab import (
    CORPORATE_VERBS, CORPORATE_NOUNS, CORPORATE_ADJECTIVES,
    BUZZWORD_PHRASES, MEETING_COMPLICATIONS
)

def generate_training_pairs():
    """Generate simple->corporate speak training pairs"""
    
    training_data = []
    
    # Simple to corporate translations
    simple_to_corporate = [
        ("let's start", "let's initiate our strategic alignment"),
        ("we need to talk", "we need to sync up and align on key deliverables"),
        ("good idea", "that's a paradigm-shifting value proposition"),
        ("I disagree", "I think we need to pivot our approach to maximize synergies"),
        ("the project failed", "the initiative didn't achieve its key performance indicators"),
        ("we're done", "we've successfully operationalized our core competencies"),
        ("this is broken", "we're experiencing suboptimal performance metrics"),
        ("fix it", "let's leverage our resources to optimize functionality"),
        ("make it work", "ensure seamless integration across all touchpoints"),
        ("I don't understand", "let's level-set on the strategic imperatives"),
        ("explain this", "provide visibility into the value drivers"),
        ("we need help", "we require additional bandwidth to scale effectively"),
        ("it's working", "we're seeing positive traction on our north star metrics"),
        ("stop doing that", "let's pivot away from that particular vector"),
        ("continue", "maintain momentum on our transformational journey"),
        ("yes", "I'm aligned with that strategic direction"),
        ("no", "I have concerns about the ROI and stakeholder buy-in"),
        ("maybe", "let's socialize this concept and circle back"),
        ("urgent", "mission-critical with significant business impact"),
        ("not important", "not a key differentiator in our current roadmap")
    ]
    
    # Generate variations
    for simple, corporate in simple_to_corporate:
        training_data.append({
            "instruction": "Convert this simple statement to corporate speak",
            "input": simple,
            "output": corporate
        })
        
    # Generate meeting complications
    simple_meetings = [
        "let's have a meeting",
        "we should discuss this",
        "can we talk about the project?",
        "I need an update",
        "when can we meet?",
        "let's review the plan",
        "we need to make a decision"
    ]
    
    for simple in simple_meetings:
        complications = []
        for _ in range(3):
            complication = random.choice(MEETING_COMPLICATIONS)
            verb = random.choice(CORPORATE_VERBS)
            noun = random.choice(CORPORATE_NOUNS)
            complications.append(f"{complication} We need to {verb} our {noun}.")
        
        training_data.append({
            "instruction": "Complicate this meeting request with corporate speak",
            "input": simple,
            "output": " ".join(complications)
        })
    
    # Generate title creations
    simple_titles = [
        "software developer",
        "manager",
        "analyst",
        "coordinator",
        "assistant",
        "engineer",
        "designer",
        "consultant"
    ]
    
    for title in simple_titles:
        adjective = random.choice(CORPORATE_ADJECTIVES)
        noun = random.choice(CORPORATE_NOUNS)
        corporate_title = f"Senior {adjective.title()} {noun.title()} {title.title()}"
        
        training_data.append({
            "instruction": "Create a corporate title for this role",
            "input": title,
            "output": corporate_title
        })
    
    # Generate buzzword salad
    for _ in range(100):
        num_buzzwords = random.randint(3, 6)
        buzzwords = random.sample(BUZZWORD_PHRASES, num_buzzwords)
        
        sentence_templates = [
            "We need to {v1} and {v2} to achieve {adj} {noun}",
            "Our {adj1} strategy involves {v1} the {noun1} while {v2} the {noun2}",
            "To {phrase1}, we must {v1} our {adj} {noun} and {phrase2}",
            "The {adj1} {noun1} requires us to {v1} and {v2} for maximum {noun2}"
        ]
        
        template = random.choice(sentence_templates)
        output = template.format(
            v1=random.choice(CORPORATE_VERBS),
            v2=random.choice(CORPORATE_VERBS),
            adj=random.choice(CORPORATE_ADJECTIVES),
            adj1=random.choice(CORPORATE_ADJECTIVES),
            noun=random.choice(CORPORATE_NOUNS),
            noun1=random.choice(CORPORATE_NOUNS),
            noun2=random.choice(CORPORATE_NOUNS),
            phrase1=buzzwords[0],
            phrase2=buzzwords[1] if len(buzzwords) > 1 else random.choice(BUZZWORD_PHRASES)
        )
        
        training_data.append({
            "instruction": "Generate corporate jargon",
            "input": "business strategy",
            "output": output
        })
    
    return training_data

def save_dataset(output_path="data/corporate_training.jsonl"):
    """Save training dataset in JSONL format"""
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    training_data = generate_training_pairs()
    
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(training_data)} training examples")
    return output_path

if __name__ == "__main__":
    save_dataset()