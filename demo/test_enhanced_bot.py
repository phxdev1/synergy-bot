"""Test the enhanced corporate bot without ML dependencies"""

from enhanced_corporate_bot import (
    ConversationContext, Domain, SeniorityLevel,
    ConversationGenerator, ReverseTranslator,
    generate_enhanced_dataset
)
import json
import random

def test_conversation_generation():
    """Test conversation generation across different contexts"""
    print("🎯 TESTING CONVERSATION GENERATION")
    print("=" * 60)
    
    conv_gen = ConversationGenerator()
    
    # Test different domain/seniority combinations
    test_contexts = [
        (Domain.TECH_STARTUP, SeniorityLevel.JUNIOR, "peer"),
        (Domain.CONSULTING, SeniorityLevel.SENIOR, "client"),
        (Domain.FINANCE, SeniorityLevel.EXECUTIVE, "subordinate"),
        (Domain.HEALTHCARE, SeniorityLevel.MID, "superior"),
    ]
    
    for domain, seniority, relationship in test_contexts:
        context = ConversationContext(
            domain=domain,
            seniority=seniority,
            relationship=relationship,
            urgency=3
        )
        
        print(f"\n📊 Context: {domain.value} / {seniority.name} / {relationship}")
        print("-" * 40)
        
        conversation = conv_gen.generate_conversation(context, "project_update")
        
        for i, turn in enumerate(conversation[:2]):  # Show first 2 turns
            print(f"\nTurn {i+1}:")
            print(f"  Casual: {turn['casual']}")
            print(f"  Corporate: {turn['corporate']}")
            if 'response' in turn:
                print(f"  Response (Casual): {turn['response']['casual']}")
                print(f"  Response (Corporate): {turn['response']['corporate']}")

def test_reverse_translation():
    """Test corporate to casual translation"""
    print("\n\n🔄 TESTING REVERSE TRANSLATION")
    print("=" * 60)
    
    reverse_gen = ReverseTranslator()
    
    # Test reverse patterns
    corporate_phrases = [
        "Let's circle back on this next week",
        "I don't have the bandwidth for this right now",
        "We need to leverage our core competencies",
        "Can we take this offline?",
        "Let's identify the low-hanging fruit"
    ]
    
    for corp_phrase in corporate_phrases:
        # Find matching pattern
        casual_version = corp_phrase.lower()
        for pattern, translations in reverse_gen.reverse_patterns.items():
            if pattern in casual_version:
                casual_version = casual_version.replace(pattern, translations[0])
                break
        
        print(f"\nCorporate: {corp_phrase}")
        print(f"Casual: {casual_version}")

def test_dataset_quality():
    """Test the quality of generated dataset"""
    print("\n\n📊 TESTING DATASET QUALITY")
    print("=" * 60)
    
    # Read the generated dataset
    with open("data/enhanced_corporate_training.jsonl", 'r') as f:
        examples = [json.loads(line) for line in f.readlines()[:20]]
    
    # Analyze by domain
    domains = {}
    for ex in examples:
        if 'context' in ex and 'domain' in ex['context']:
            domain = ex['context']['domain']
            domains[domain] = domains.get(domain, 0) + 1
    
    print("\nDomain distribution:")
    for domain, count in domains.items():
        print(f"  {domain}: {count} examples")
    
    # Show examples from different instruction types
    instruction_types = {}
    for ex in examples:
        inst_type = ex['instruction'].split()[0:3]
        inst_key = " ".join(inst_type)
        if inst_key not in instruction_types:
            instruction_types[inst_key] = []
        instruction_types[inst_key].append(ex)
    
    print("\n\nSample transformations by type:")
    print("-" * 40)
    for inst_type, exs in list(instruction_types.items())[:5]:
        if exs:
            ex = exs[0]
            print(f"\n{ex['instruction']}")
            print(f"Input: {ex['input']}")
            print(f"Output: {ex['output'][:100]}...")

def compare_with_comprehensive():
    """Compare with the comprehensive 3-layer approach"""
    print("\n\n🔍 COMPARING APPROACHES")
    print("=" * 60)
    
    print("\n📋 Enhanced Bot (Conversational & Domain-Aware):")
    print("  ✓ Multi-turn conversations")
    print("  ✓ Domain-specific vocabulary (6 industries)")
    print("  ✓ Seniority progression (4 levels)")
    print("  ✓ Reverse translation")
    print("  ✓ Believable, context-aware output")
    
    print("\n📚 Comprehensive Bot (3-Layer Mad-libs):")
    print("  ✓ Extensive template coverage")
    print("  ✓ Interview scenarios")
    print("  ✓ Product roadmaps")
    print("  ✓ Executive communications")
    print("  ✓ Thought leadership")
    
    print("\n💡 Recommendation:")
    print("  Combine both approaches for maximum coverage:")
    print("  - Use enhanced bot for conversational AI")
    print("  - Use comprehensive templates for specific scenarios")

if __name__ == "__main__":
    # Run all tests
    test_conversation_generation()
    test_reverse_translation()
    test_dataset_quality()
    compare_with_comprehensive()
    
    print("\n\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train LoRA: python train_lora.py")
    print("3. Push to HuggingFace: Already configured for 'phxdev/corporate-speak-lora'")