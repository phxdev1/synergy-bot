"""Combined corporate bot - merging conversational AI with comprehensive templates"""

import json
import random
from enhanced_corporate_bot import (
    Domain, SeniorityLevel, ConversationContext,
    ConversationGenerator, ReverseTranslator,
    DOMAIN_VOCAB, SENIORITY_PATTERNS
)
from test_synergy import (
    INTERVIEW_TEMPLATES, ROADMAP_TEMPLATES, WEBSITE_TEMPLATES,
    VISION_TEMPLATES, EXECUTIVE_TEMPLATES,
    generate_from_template as generate_comprehensive_template
)

class CombinedCorporateBot:
    """Combines conversational and template-based approaches"""
    
    def __init__(self):
        self.conversation_gen = ConversationGenerator()
        self.reverse_translator = ReverseTranslator()
        
    def generate_combined_dataset(self, num_examples: int = 3000) -> list:
        """Generate dataset combining both approaches"""
        training_data = []
        
        # 40% conversational chains
        conv_examples = int(num_examples * 0.4)
        training_data.extend(self._generate_conversational_examples(conv_examples))
        
        # 30% specialized scenarios (interviews, roadmaps, etc.)
        specialized_examples = int(num_examples * 0.3)
        training_data.extend(self._generate_specialized_examples(specialized_examples))
        
        # 20% reverse translations
        reverse_examples = int(num_examples * 0.2)
        training_data.extend(self._generate_reverse_examples(reverse_examples))
        
        # 10% hybrid examples (combining approaches)
        hybrid_examples = int(num_examples * 0.1)
        training_data.extend(self._generate_hybrid_examples(hybrid_examples))
        
        # Ensure exact count
        random.shuffle(training_data)
        return training_data[:num_examples]
    
    def _generate_conversational_examples(self, count: int) -> list:
        """Generate conversation-based examples"""
        examples = []
        domains = list(Domain)
        seniority_levels = list(SeniorityLevel)
        relationships = ["peer", "superior", "subordinate", "client"]
        
        for _ in range(count // 10):  # Generate batches
            context = ConversationContext(
                domain=random.choice(domains),
                seniority=random.choice(seniority_levels),
                relationship=random.choice(relationships),
                urgency=random.randint(1, 5)
            )
            
            conversation_type = random.choice(["project_update", "meeting_request", "feedback_delivery"])
            conversation = self.conversation_gen.generate_conversation(context, conversation_type)
            
            for turn in conversation:
                examples.append({
                    "instruction": f"Transform to {context.domain.value} corporate speak (seniority: {context.seniority.name})",
                    "input": turn["casual"],
                    "output": turn["corporate"],
                    "context": {
                        "domain": context.domain.value,
                        "seniority": context.seniority.value,
                        "relationship": context.relationship,
                        "conversation_type": conversation_type
                    }
                })
                
                if "response" in turn:
                    examples.append({
                        "instruction": f"Generate a professional response in {context.domain.value} style",
                        "input": turn["response"]["casual"],
                        "output": turn["response"]["corporate"],
                        "context": {
                            "domain": context.domain.value,
                            "seniority": context.seniority.value,
                            "relationship": context.relationship,
                            "conversation_type": conversation_type
                        }
                    })
        
        return examples
    
    def _generate_specialized_examples(self, count: int) -> list:
        """Generate template-based specialized scenarios"""
        examples = []
        
        # Interview scenarios
        interview_questions = [
            "tell me about yourself",
            "why do you want this job?",
            "what are your strengths?",
            "describe a challenge you faced",
            "where do you see yourself in 5 years?"
        ]
        
        for _ in range(count // 5):
            for question in interview_questions:
                # Add domain context to interview responses
                domain = random.choice(list(Domain))
                domain_vocab = DOMAIN_VOCAB[domain]
                
                # Generate response with domain flavor
                template = random.choice(INTERVIEW_TEMPLATES["candidate_response"])
                response = generate_comprehensive_template(template)
                
                # Inject domain-specific terms
                response = response.replace("corporate_noun", random.choice(domain_vocab["nouns"]))
                
                examples.append({
                    "instruction": f"Respond to interview question as {domain.value} professional",
                    "input": question,
                    "output": response,
                    "context": {
                        "domain": domain.value,
                        "scenario": "interview"
                    }
                })
        
        # Product roadmap communications
        for _ in range(count // 5):
            domain = random.choice(list(Domain))
            for template_type in ["feature_announcement", "roadmap_vision"]:
                template = random.choice(ROADMAP_TEMPLATES[template_type])
                output = generate_comprehensive_template(template)
                
                examples.append({
                    "instruction": f"Create {template_type.replace('_', ' ')} for {domain.value} company",
                    "input": f"announce new {template_type}",
                    "output": output,
                    "context": {
                        "domain": domain.value,
                        "scenario": "product_communication"
                    }
                })
        
        # Executive communications with seniority awareness
        for _ in range(count // 5):
            seniority = random.choice([SeniorityLevel.SENIOR, SeniorityLevel.EXECUTIVE])
            template = random.choice(EXECUTIVE_TEMPLATES["board_presentation"])
            output = generate_comprehensive_template(template)
            
            # Add seniority-appropriate tone
            if seniority == SeniorityLevel.EXECUTIVE:
                output = output.replace("We've", "We have").replace("We're", "We are")
            
            examples.append({
                "instruction": f"Create executive communication (level: {seniority.name})",
                "input": "quarterly board update",
                "output": output,
                "context": {
                    "seniority": seniority.value,
                    "scenario": "executive_communication"
                }
            })
        
        return examples
    
    def _generate_reverse_examples(self, count: int) -> list:
        """Generate reverse translation examples"""
        examples = []
        reverse_pairs = self.reverse_translator.generate_reverse_pairs(count)
        
        # Add domain context to some reverse translations
        for i, pair in enumerate(reverse_pairs):
            if i % 3 == 0:  # Every third example
                domain = random.choice(list(Domain))
                pair["instruction"] = f"Translate {domain.value} corporate speak to plain English"
                pair["context"] = {"domain": domain.value}
            examples.append(pair)
        
        return examples
    
    def _generate_hybrid_examples(self, count: int) -> list:
        """Generate examples combining multiple approaches"""
        examples = []
        
        for _ in range(count):
            # Create multi-step transformations
            domain = random.choice(list(Domain))
            seniority = random.choice(list(SeniorityLevel))
            
            # Start with casual multi-part message
            casual_parts = [
                "hey team",
                "quick update on the project",
                "we hit some issues but found solutions",
                "let me know if you need details"
            ]
            
            # Transform each part according to context
            corporate_parts = []
            for part in casual_parts:
                if "hey" in part:
                    corporate_parts.append("Good morning team,")
                elif "quick update" in part:
                    corporate_parts.append(f"I wanted to provide a brief update on our {random.choice(DOMAIN_VOCAB[domain]['nouns'])}")
                elif "issues" in part:
                    corporate_parts.append(f"While we encountered some {random.choice(DOMAIN_VOCAB[domain]['concerns'])}, we've identified viable solutions")
                elif "let me know" in part:
                    corporate_parts.append("Please don't hesitate to reach out if you require additional details")
            
            examples.append({
                "instruction": f"Transform multi-part message to {domain.value} corporate style (level: {seniority.name})",
                "input": "\n".join(casual_parts),
                "output": "\n".join(corporate_parts),
                "context": {
                    "domain": domain.value,
                    "seniority": seniority.value,
                    "message_type": "multi_part_update"
                }
            })
        
        return examples

def generate_ultimate_dataset(output_path: str = "data/ultimate_corporate_training.jsonl", 
                            num_examples: int = 5000):
    """Generate the ultimate combined dataset"""
    bot = CombinedCorporateBot()
    training_data = bot.generate_combined_dataset(num_examples)
    
    # Save dataset
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    # Print statistics
    print(f"🚀 Generated {len(training_data)} ultimate training examples")
    print("\n📊 Dataset Composition:")
    print("  • 40% Conversational chains (multi-turn, context-aware)")
    print("  • 30% Specialized scenarios (interviews, roadmaps, executive)")
    print("  • 20% Reverse translations (corporate → casual)")
    print("  • 10% Hybrid examples (multi-part, complex transformations)")
    
    # Analyze coverage
    domains = {}
    scenarios = {}
    seniority = {}
    
    for item in training_data:
        if 'context' in item:
            if 'domain' in item['context']:
                d = item['context']['domain']
                domains[d] = domains.get(d, 0) + 1
            if 'scenario' in item['context']:
                s = item['context']['scenario']
                scenarios[s] = scenarios.get(s, 0) + 1
            if 'seniority' in item['context']:
                sn = item['context']['seniority']
                seniority[sn] = seniority.get(sn, 0) + 1
    
    print("\n🏢 Domain Coverage:")
    for d, count in sorted(domains.items()):
        print(f"  {d}: {count} examples")
    
    print("\n🎯 Scenario Coverage:")
    for s, count in sorted(scenarios.items()):
        print(f"  {s}: {count} examples")
    
    print("\n📈 Seniority Levels:")
    for level, count in sorted(seniority.items()):
        level_name = ["", "JUNIOR", "MID", "SENIOR", "EXECUTIVE"][level] if isinstance(level, int) else str(level)
        print(f"  {level_name}: {count} examples")
    
    # Show diverse examples
    print("\n✨ Sample Transformations:")
    print("=" * 60)
    sample_indices = random.sample(range(len(training_data)), min(8, len(training_data)))
    for i, idx in enumerate(sample_indices, 1):
        item = training_data[idx]
        print(f"\n{i}. {item['instruction']}")
        print(f"   Input: {item['input'][:80]}{'...' if len(item['input']) > 80 else ''}")
        print(f"   Output: {item['output'][:80]}{'...' if len(item['output']) > 80 else ''}")
        if 'context' in item:
            print(f"   Context: {item['context']}")
    
    return output_path

if __name__ == "__main__":
    # Generate the ultimate dataset
    generate_ultimate_dataset(num_examples=5000)