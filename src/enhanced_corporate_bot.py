"""Enhanced corporate speak bot with realistic conversations and domain adaptation"""

import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class Domain(Enum):
    TECH_STARTUP = "tech_startup"
    CONSULTING = "consulting"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"

class SeniorityLevel(Enum):
    JUNIOR = 1
    MID = 2
    SENIOR = 3
    EXECUTIVE = 4

@dataclass
class ConversationContext:
    domain: Domain
    seniority: SeniorityLevel
    relationship: str  # "peer", "superior", "subordinate", "client"
    urgency: int  # 1-5 scale
    
# Domain-specific vocabularies
DOMAIN_VOCAB = {
    Domain.TECH_STARTUP: {
        "verbs": ["iterate", "pivot", "scale", "disrupt", "bootstrap", "ship", "deploy"],
        "nouns": ["MVP", "runway", "burn rate", "user story", "sprint", "backlog", "feature"],
        "phrases": ["move fast", "fail fast", "growth hack", "product-market fit"],
        "concerns": ["technical debt", "user adoption", "feature velocity", "platform stability"]
    },
    Domain.CONSULTING: {
        "verbs": ["leverage", "synthesize", "socialize", "cascade", "operationalize"],
        "nouns": ["framework", "methodology", "best practices", "value proposition", "ROI"],
        "phrases": ["at the end of the day", "net-net", "boil the ocean", "low-hanging fruit"],
        "concerns": ["client expectations", "project scope", "billable hours", "deliverable quality"]
    },
    Domain.FINANCE: {
        "verbs": ["optimize", "hedge", "diversify", "consolidate", "audit", "forecast"],
        "nouns": ["portfolio", "compliance", "risk profile", "margins", "liquidity", "yield"],
        "phrases": ["risk-adjusted returns", "market conditions", "regulatory requirements"],
        "concerns": ["market volatility", "regulatory compliance", "profit margins", "audit findings"]
    },
    Domain.HEALTHCARE: {
        "verbs": ["standardize", "document", "coordinate", "comply", "streamline"],
        "nouns": ["protocols", "outcomes", "patient care", "compliance", "quality metrics"],
        "phrases": ["patient-centered", "evidence-based", "HIPAA compliant", "clinical excellence"],
        "concerns": ["patient satisfaction", "regulatory compliance", "care coordination", "documentation"]
    },
    Domain.RETAIL: {
        "verbs": ["optimize", "personalize", "convert", "engage", "streamline", "fulfill"],
        "nouns": ["customer journey", "conversion rate", "inventory", "touchpoints", "experience", "channels"],
        "phrases": ["omnichannel approach", "customer-centric", "seamless experience", "data-driven insights"],
        "concerns": ["inventory management", "customer satisfaction", "supply chain", "seasonal fluctuations"]
    },
    Domain.MANUFACTURING: {
        "verbs": ["optimize", "streamline", "standardize", "automate", "implement", "maintain"],
        "nouns": ["throughput", "quality control", "supply chain", "efficiency", "capacity", "lean processes"],
        "phrases": ["continuous improvement", "just-in-time", "six sigma", "operational excellence"],
        "concerns": ["production delays", "quality issues", "supply chain disruptions", "safety compliance"]
    }
}

# Seniority-based language patterns
SENIORITY_PATTERNS = {
    SeniorityLevel.JUNIOR: {
        "confidence": ["I think", "maybe we could", "would it be possible to"],
        "questions": ["Could you help me understand", "I was wondering if", "Is it okay if"],
        "updates": ["I'm working on", "I've completed", "I'm stuck on"]
    },
    SeniorityLevel.MID: {
        "confidence": ["I suggest", "we should consider", "it would be beneficial to"],
        "questions": ["What are your thoughts on", "Have we considered", "Could we explore"],
        "updates": ["I'm making progress on", "We've identified", "The team is addressing"]
    },
    SeniorityLevel.SENIOR: {
        "confidence": ["I recommend", "we need to", "it's critical that we"],
        "questions": ["What's our strategy for", "How are we positioned on", "What's the impact of"],
        "updates": ["We're on track with", "I'm driving", "We've successfully"]
    },
    SeniorityLevel.EXECUTIVE: {
        "confidence": ["We will", "I expect", "it's imperative that"],
        "questions": ["What's our position on", "How does this align with", "What's the business case"],
        "updates": ["We've achieved", "I'm pleased to report", "We're leading"]
    }
}

class ConversationGenerator:
    def __init__(self):
        self.conversation_templates = self._load_conversation_templates()
        
    def _load_conversation_templates(self) -> Dict[str, List[Dict]]:
        """Load realistic conversation flow templates"""
        return {
            "project_update": [
                {
                    "casual": "hey, how's the project going?",
                    "corporate": "I wanted to touch base on the {project_noun} project status.",
                    "response_casual": "it's going okay, hit a few snags",
                    "response_corporate": "We're making steady progress, though we've encountered some {concern} that we're actively addressing."
                },
                {
                    "casual": "what kind of problems?",
                    "corporate": "Could you elaborate on the challenges you're facing?",
                    "response_casual": "the timeline is tight and we need more people",
                    "response_corporate": "The current timeline is aggressive given our resource constraints. We may need to discuss augmenting the team or adjusting our deliverable schedule."
                }
            ],
            "meeting_request": [
                {
                    "casual": "got time to chat?",
                    "corporate": "Do you have availability for a brief discussion?",
                    "response_casual": "sure, when?",
                    "response_corporate": "I can make myself available. What timeframe works best for you?"
                },
                {
                    "casual": "how about tomorrow at 2?",
                    "corporate": "Would tomorrow at 2 PM work with your schedule?",
                    "response_casual": "works for me",
                    "response_corporate": "That works perfectly. I'll send a calendar invite shortly."
                }
            ],
            "feedback_delivery": [
                {
                    "casual": "nice work on the presentation",
                    "corporate": "I wanted to commend you on the {deliverable} presentation. It was well-structured and effectively communicated our key points.",
                    "response_casual": "thanks! glad it went well",
                    "response_corporate": "Thank you for the feedback. I'm pleased it resonated with the audience."
                },
                {
                    "casual": "maybe add more data next time",
                    "corporate": "For future iterations, incorporating additional data points could strengthen our narrative.",
                    "response_casual": "good idea, I'll do that",
                    "response_corporate": "That's valuable input. I'll ensure we include more quantitative support in upcoming presentations."
                }
            ]
        }
    
    def generate_conversation(self, context: ConversationContext, 
                            conversation_type: str = "project_update") -> List[Dict]:
        """Generate a believable conversation chain"""
        template = self.conversation_templates.get(conversation_type, self.conversation_templates["project_update"])
        domain_vocab = DOMAIN_VOCAB[context.domain]
        seniority_patterns = SENIORITY_PATTERNS[context.seniority]
        
        conversation = []
        for turn in template:
            # Adapt the corporate version to domain and seniority
            corporate_msg = turn["corporate"]
            if "{project_noun}" in corporate_msg:
                corporate_msg = corporate_msg.format(project_noun=random.choice(domain_vocab["nouns"]))
            if "{concern}" in corporate_msg:
                corporate_msg = corporate_msg.format(concern=random.choice(domain_vocab["concerns"]))
            if "{deliverable}" in corporate_msg:
                corporate_msg = corporate_msg.format(deliverable=random.choice(domain_vocab["nouns"]))
            
            # Add seniority-appropriate prefix
            if context.seniority == SeniorityLevel.JUNIOR and random.random() > 0.5:
                corporate_msg = f"{random.choice(seniority_patterns['confidence'])} {corporate_msg.lower()}"
            
            entry = {
                "casual": turn["casual"],
                "corporate": corporate_msg,
                "context": {
                    "domain": context.domain.value,
                    "seniority": context.seniority.value,
                    "relationship": context.relationship
                }
            }
            
            if "response_casual" in turn:
                response_corporate = turn["response_corporate"]
                if "{concern}" in response_corporate:
                    response_corporate = response_corporate.format(
                        concern=random.choice(domain_vocab["concerns"])
                    )
                
                entry["response"] = {
                    "casual": turn["response_casual"],
                    "corporate": response_corporate
                }
            
            conversation.append(entry)
        
        return conversation

class ReverseTranslator:
    """Generate reverse translation pairs (corporate -> casual)"""
    
    def __init__(self):
        self.reverse_patterns = {
            "let's circle back": ["let's talk later", "we'll discuss this later", "let's revisit this"],
            "bandwidth": ["time", "capacity", "availability"],
            "synergize": ["work together", "collaborate", "combine efforts"],
            "leverage": ["use", "utilize", "take advantage of"],
            "touch base": ["check in", "talk", "connect"],
            "deep dive": ["detailed look", "thorough review", "close examination"],
            "move the needle": ["make progress", "make a difference", "have impact"],
            "low-hanging fruit": ["easy wins", "quick fixes", "simple tasks"],
            "take this offline": ["discuss privately", "talk later", "handle separately"]
        }
    
    def generate_reverse_pairs(self, num_examples: int = 100) -> List[Dict]:
        """Generate corporate -> casual translation pairs"""
        pairs = []
        
        corporate_phrases = [
            "Let's leverage our core competencies to drive value",
            "We need to circle back on the deliverables",
            "I don't have the bandwidth for additional asks",
            "Let's take a deep dive into the metrics",
            "We should identify the low-hanging fruit",
            "Can we synergize our efforts on this initiative?",
            "This will really move the needle on our KPIs",
            "Let's touch base early next week",
            "We should take this conversation offline",
            "I'll socialize this with the stakeholders"
        ]
        
        casual_translations = [
            "Let's use what we're good at to help",
            "We need to talk about what's due",
            "I don't have time for more work",
            "Let's look closely at the numbers",
            "We should do the easy stuff first",
            "Can we work together on this?",
            "This will really help our goals",
            "Let's talk early next week",
            "Let's discuss this privately",
            "I'll share this with everyone involved"
        ]
        
        for corp, casual in zip(corporate_phrases, casual_translations):
            pairs.append({
                "instruction": "Translate this corporate speak to plain English",
                "input": corp,
                "output": casual
            })
        
        # Generate more variations
        for pattern, translations in self.reverse_patterns.items():
            for _ in range(num_examples // len(self.reverse_patterns)):
                corporate_sentence = self._generate_sentence_with_pattern(pattern)
                casual_sentence = corporate_sentence
                for translation in random.sample(translations, 1):
                    casual_sentence = casual_sentence.replace(pattern, translation)
                
                pairs.append({
                    "instruction": "Translate this corporate speak to plain English",
                    "input": corporate_sentence,
                    "output": casual_sentence
                })
        
        return pairs
    
    def _generate_sentence_with_pattern(self, pattern: str) -> str:
        """Generate a sentence containing the corporate pattern"""
        templates = [
            f"We should {pattern} on this matter",
            f"Let's {pattern} regarding the project",
            f"I'd like to {pattern} about your concerns",
            f"Can we {pattern} on the proposal?",
            f"It would be beneficial to {pattern}"
        ]
        return random.choice(templates).replace(f"{pattern}", pattern)

def generate_enhanced_dataset(num_examples: int = 1000) -> str:
    """Generate the enhanced training dataset"""
    training_data = []
    
    # Initialize generators
    conv_gen = ConversationGenerator()
    reverse_gen = ReverseTranslator()
    
    # Generate conversations for different contexts
    domains = list(Domain)
    seniority_levels = list(SeniorityLevel)
    relationships = ["peer", "superior", "subordinate", "client"]
    
    # 1. Generate conversation chains (40% of dataset)
    for _ in range(int(num_examples * 0.4)):
        context = ConversationContext(
            domain=random.choice(domains),
            seniority=random.choice(seniority_levels),
            relationship=random.choice(relationships),
            urgency=random.randint(1, 5)
        )
        
        conversation_type = random.choice(["project_update", "meeting_request", "feedback_delivery"])
        conversation = conv_gen.generate_conversation(context, conversation_type)
        
        # Add each turn as a training example
        for turn in conversation:
            training_data.append({
                "instruction": f"Transform to {context.domain.value} corporate speak (seniority: {context.seniority.name})",
                "input": turn["casual"],
                "output": turn["corporate"],
                "context": turn.get("context", {})
            })
            
            if "response" in turn:
                training_data.append({
                    "instruction": f"Generate a professional response in {context.domain.value} style",
                    "input": turn["response"]["casual"],
                    "output": turn["response"]["corporate"],
                    "context": turn.get("context", {})
                })
    
    # 2. Generate reverse translations (30% of dataset)
    reverse_pairs = reverse_gen.generate_reverse_pairs(int(num_examples * 0.3))
    training_data.extend(reverse_pairs)
    
    # 3. Generate domain-specific transformations (30% of dataset)
    for _ in range(int(num_examples * 0.3)):
        domain = random.choice(domains)
        seniority = random.choice(seniority_levels)
        domain_vocab = DOMAIN_VOCAB[domain]
        
        casual_phrases = [
            "we need to fix this",
            "the project is delayed",
            "good job on this",
            "can you help?",
            "I disagree",
            "this isn't working"
        ]
        
        casual = random.choice(casual_phrases)
        
        # Build corporate version using domain vocabulary
        if casual == "we need to fix this":
            corporate = f"We need to {random.choice(domain_vocab['verbs'])} our approach to address these {random.choice(domain_vocab['concerns'])}"
        elif casual == "the project is delayed":
            corporate = f"The {random.choice(domain_vocab['nouns'])} timeline has shifted due to {random.choice(domain_vocab['concerns'])}"
        elif casual == "good job on this":
            corporate = f"Excellent work {random.choice(['implementing', 'delivering', 'executing'])} the {random.choice(domain_vocab['nouns'])}"
        elif casual == "can you help?":
            corporate = f"Could you provide support in {random.choice(domain_vocab['verbs'])}ing our {random.choice(domain_vocab['nouns'])}?"
        elif casual == "I disagree":
            corporate = f"I have a different perspective on how we should {random.choice(domain_vocab['verbs'])} this {random.choice(domain_vocab['nouns'])}"
        else:  # "this isn't working"
            corporate = f"We're experiencing challenges with the current {random.choice(domain_vocab['nouns'])} approach"
        
        # Add seniority-appropriate modifiers
        if seniority == SeniorityLevel.JUNIOR:
            corporate = f"{random.choice(SENIORITY_PATTERNS[seniority]['confidence'])} {corporate.lower()}"
        
        training_data.append({
            "instruction": f"Transform to {domain.value} corporate speak",
            "input": casual,
            "output": corporate,
            "context": {
                "domain": domain.value,
                "seniority": seniority.value
            }
        })
    
    # Save the dataset
    output_path = "data/enhanced_corporate_training.jsonl"
    with open(output_path, 'w') as f:
        for item in training_data[:num_examples]:  # Trim to exact number
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {num_examples} enhanced training examples")
    print("\nSample outputs:")
    for i, item in enumerate(random.sample(training_data[:num_examples], 5)):
        print(f"\n{i+1}. Domain: {item.get('context', {}).get('domain', 'general')}")
        print(f"   '{item['input']}' → '{item['output']}'")
    
    return output_path

if __name__ == "__main__":
    generate_enhanced_dataset(1000)