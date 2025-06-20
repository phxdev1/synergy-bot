"""
3-Layer Corporate Communication Mad-libs Generator
Layer 1: Basic corporate communication (emails, meetings, etc.)
Layer 2: Specialized scenarios (interviews, roadmaps, websites)
Layer 3: Meta-corporate content (culture, vision, thought leadership)
"""

import json
import random
from itertools import product, combinations

# ============================================================================
# LAYER 1: FOUNDATIONAL CORPORATE VOCABULARY (Enhanced from previous)
# ============================================================================

# Core vocabulary from Layer 1
CORPORATE_VERBS = [
    "optimize", "leverage", "streamline", "align", "synergize", "enhance", 
    "facilitate", "coordinate", "prioritize", "implement", "execute",
    "strategize", "collaborate", "communicate", "deliver", "achieve"
]

CORPORATE_NOUNS = [
    "deliverables", "objectives", "stakeholders", "initiatives", "touchpoints",
    "outcomes", "metrics", "processes", "workflows", "strategies", "solutions",
    "opportunities", "requirements", "expectations", "capabilities", "resources"
]

CORPORATE_GERUNDS = [
    "working on", "reviewing", "updating", "managing", "tracking",
    "coordinating", "organizing", "planning", "developing", "improving"
]

ACTION_VERBS = [
    "connect", "sync", "touch base", "circle back", "follow up", "check in",
    "reconnect", "reach out", "loop back", "catch up", "ping", "schedule time"
]

# ============================================================================
# LAYER 2: SPECIALIZED SCENARIO VOCABULARY
# ============================================================================

# Interview and recruiting vocabulary
INTERVIEW_SKILLS = [
    "leadership capabilities", "technical expertise", "problem-solving abilities",
    "communication skills", "analytical thinking", "strategic mindset",
    "collaborative approach", "innovation potential", "cultural fit", "growth mindset"
]

INTERVIEW_QUALITIES = [
    "exceptional", "strong", "demonstrated", "proven", "impressive", "solid",
    "outstanding", "remarkable", "consistent", "dynamic"
]

# Product and roadmap vocabulary
PRODUCT_FEATURES = [
    "user experience enhancements", "performance optimizations", "security improvements",
    "scalability features", "integration capabilities", "automation tools",
    "analytics dashboards", "mobile functionality", "API developments", "cloud infrastructure"
]

PRODUCT_BENEFITS = [
    "increased efficiency", "enhanced user satisfaction", "improved ROI", 
    "reduced operational costs", "accelerated time-to-market", "competitive advantage",
    "scalable growth", "streamlined workflows", "data-driven insights", "seamless integration"
]

TIMELINE_PHASES = [
    "Q1 delivery", "Q2 rollout", "Q3 optimization", "Q4 evaluation",
    "immediate implementation", "phased approach", "gradual deployment", "pilot program"
]

# Website and marketing vocabulary
MARKETING_BENEFITS = [
    "transform your business", "drive meaningful results", "unlock new opportunities",
    "accelerate growth", "maximize efficiency", "optimize performance",
    "enhance customer experience", "deliver exceptional value", "achieve sustainable success"
]

MARKETING_FEATURES = [
    "cutting-edge technology", "industry-leading solutions", "comprehensive platform",
    "intuitive interface", "robust architecture", "scalable infrastructure",
    "advanced analytics", "seamless integration", "enterprise-grade security"
]

# ============================================================================
# LAYER 2: SPECIALIZED SCENARIO TEMPLATES
# ============================================================================

# Interview templates (both interviewer and candidate perspectives)
INTERVIEW_TEMPLATES = {
    "interviewer_opening": [
        "We're excited to discuss how your {interview_skills} align with our {corporate_noun}",
        "I'd love to learn more about your experience {gerund} {corporate_noun}",
        "Can you walk me through your approach to {gerund} {corporate_noun}?",
        "Tell me about a time when you {corporate_verb}ed {corporate_noun} successfully"
    ],
    "candidate_response": [
        "I'm passionate about {gerund} {corporate_noun} to {business_outcome}",
        "In my previous role, I focused on {gerund} {corporate_noun} which resulted in {business_outcome}",
        "I believe my {interview_skills} would enable me to {corporate_verb} your {corporate_noun}",
        "I'm excited about the opportunity to {corporate_verb} {corporate_noun} and drive {business_outcome}"
    ],
    "interview_followup": [
        "Thank you for taking the time to discuss your {interview_skills}",
        "We'll {action_verb} {time_expression} regarding next steps in our process",
        "Your experience {gerund} {corporate_noun} is exactly what we're looking for",
        "We're impressed with your {interview_qualities} background in {corporate_noun}"
    ]
}

# Product roadmap templates
ROADMAP_TEMPLATES = {
    "feature_announcement": [
        "We're excited to introduce {product_features} that will deliver {product_benefits}",
        "Our {timeline_phases} will focus on {gerund} {product_features} to achieve {product_benefits}",
        "The upcoming release includes {product_features} designed to {corporate_verb} {product_benefits}",
        "We're {gerund} {product_features} that will {corporate_verb} your {corporate_noun}"
    ],
    "roadmap_vision": [
        "Our product roadmap is focused on {gerund} {product_features} to transform how you {corporate_verb} {corporate_noun}",
        "Over the next quarter, we'll be {gerund} key {product_features} that deliver {product_benefits}",
        "We're committed to {gerund} innovative {product_features} that {corporate_verb} your {business_outcome}",
        "The roadmap prioritizes {product_features} to ensure {product_benefits} for our stakeholders"
    ],
    "timeline_communication": [
        "We're targeting {timeline_phases} for the rollout of {product_features}",
        "The {timeline_phases} will include comprehensive {product_features} implementation",
        "Our {timeline_phases} focuses on {gerund} {product_features} with {stakeholder_group}",
        "We'll {action_verb} with updates as we progress through {timeline_phases}"
    ]
}

# Website and marketing templates
WEBSITE_TEMPLATES = {
    "value_proposition": [
        "Our {marketing_features} help organizations {corporate_verb} their {corporate_noun} to {marketing_benefits}",
        "We specialize in {gerund} {marketing_features} that enable businesses to {marketing_benefits}",
        "Transform your {corporate_noun} with our {marketing_features} designed to {marketing_benefits}",
        "Discover how our {marketing_features} can {corporate_verb} your {corporate_noun} and {marketing_benefits}"
    ],
    "feature_description": [
        "Our {marketing_features} provide the {corporate_noun} you need to {marketing_benefits}",
        "Experience {marketing_features} that {corporate_verb} your {corporate_noun} while {gerund} {business_outcome}",
        "Built with {marketing_features}, our platform helps you {corporate_verb} {corporate_noun} efficiently",
        "Leverage our {marketing_features} to {corporate_verb} {corporate_noun} and achieve {business_outcome}"
    ],
    "call_to_action": [
        "Ready to {corporate_verb} your {corporate_noun}? Let's {action_verb} to discuss your {corporate_noun}",
        "Discover how we can help you {marketing_benefits} - {action_verb} with our team today",
        "Take the next step in {gerund} your {corporate_noun} - {action_verb} for a consultation",
        "Start {gerund} better {corporate_noun} today - {action_verb} to learn more about our {marketing_features}"
    ]
}

# ============================================================================
# LAYER 3: META-CORPORATE VOCABULARY & TEMPLATES
# ============================================================================

# Vision and culture vocabulary
VISION_CONCEPTS = [
    "industry transformation", "sustainable innovation", "global impact", "customer-centricity",
    "operational excellence", "digital transformation", "collaborative ecosystem", "future-ready solutions",
    "purpose-driven growth", "stakeholder value creation"
]

CULTURE_VALUES = [
    "integrity", "innovation", "collaboration", "excellence", "accountability",
    "transparency", "agility", "customer focus", "continuous learning", "inclusive leadership"
]

THOUGHT_LEADERSHIP_TOPICS = [
    "digital transformation trends", "future of work", "sustainable business practices",
    "innovation methodologies", "leadership paradigms", "customer experience evolution",
    "data-driven decision making", "organizational agility", "technology integration"
]

# Executive and board communication
EXECUTIVE_CONCERNS = [
    "market positioning", "competitive landscape", "operational efficiency", "revenue growth",
    "risk management", "stakeholder engagement", "strategic alignment", "digital readiness",
    "talent acquisition", "sustainability initiatives"
]

BOARD_METRICS = [
    "quarterly performance indicators", "strategic milestone achievements", "market share growth",
    "operational efficiency metrics", "customer satisfaction scores", "employee engagement levels",
    "financial performance ratios", "risk assessment outcomes", "innovation pipeline progress"
]

# ============================================================================
# LAYER 3: META-CORPORATE TEMPLATES
# ============================================================================

VISION_TEMPLATES = {
    "mission_statement": [
        "We exist to {corporate_verb} {vision_concepts} through {gerund} innovative {corporate_noun}",
        "Our mission is {gerund} {vision_concepts} by {corporate_verb}ing {corporate_noun} for {stakeholder_group}",
        "We're committed to {gerund} {vision_concepts} that {corporate_verb} {business_outcome} globally",
        "We empower organizations to achieve {vision_concepts} through our {marketing_features}"
    ],
    "culture_description": [
        "Our culture is built on {culture_values} and {gerund} {corporate_noun} that drive {business_outcome}",
        "We foster {culture_values} while {gerund} an environment where {stakeholder_group} can {corporate_verb} {corporate_noun}",
        "At our core, we believe in {culture_values} and {gerund} {vision_concepts} through collaborative {corporate_noun}",
        "We champion {culture_values} by {gerund} {corporate_noun} that reflect our commitment to {vision_concepts}"
    ],
    "thought_leadership": [
        "The future of {thought_leadership_topics} requires {gerund} {corporate_noun} that {corporate_verb} {vision_concepts}",
        "As leaders in {thought_leadership_topics}, we're {gerund} innovative approaches to {corporate_verb} {corporate_noun}",
        "Our perspective on {thought_leadership_topics} is shaped by our experience {gerund} {corporate_noun}",
        "We believe {thought_leadership_topics} will transform how organizations {corporate_verb} their {corporate_noun}"
    ]
}

EXECUTIVE_TEMPLATES = {
    "board_presentation": [
        "Our Q{quarter} performance demonstrates strong progress in {gerund} {board_metrics}",
        "We've successfully {corporate_verb}ed our {executive_concerns} resulting in improved {board_metrics}",
        "The board should note our {achievement_adjective} results in {gerund} {executive_concerns}",
        "Our strategic focus on {executive_concerns} has delivered {achievement_adjective} {board_metrics}"
    ],
    "investor_communication": [
        "We're well-positioned to {corporate_verb} {executive_concerns} and deliver sustained {board_metrics}",
        "Our investment in {gerund} {corporate_noun} supports our long-term {executive_concerns} strategy",
        "We continue {gerund} {executive_concerns} while maintaining strong {board_metrics}",
        "The market opportunity for {corporate_noun} aligns with our {executive_concerns} priorities"
    ],
    "annual_report": [
        "This year's achievements in {gerund} {executive_concerns} demonstrate our commitment to {vision_concepts}",
        "We've made significant progress {gerund} {board_metrics} while advancing our {vision_concepts} agenda",
        "Our focus on {executive_concerns} has enabled us to {corporate_verb} industry-leading {board_metrics}",
        "Looking ahead, we'll continue {gerund} {vision_concepts} through strategic {corporate_noun} initiatives"
    ]
}

# ============================================================================
# ENHANCED VOCABULARY LISTS (Building on all layers)
# ============================================================================

PERFORMANCE_ADJECTIVES = [
    "outstanding", "solid", "consistent", "impressive", "strong", "effective",
    "exceptional", "remarkable", "transformational", "industry-leading"
]

BUSINESS_OUTCOMES = [
    "sustainable growth", "operational excellence", "customer satisfaction", 
    "market leadership", "competitive advantage", "organizational efficiency",
    "digital transformation", "innovation acceleration", "stakeholder value"
]

PROJECT_STATUS = [
    "on track", "progressing well", "ahead of schedule", "meeting expectations",
    "moving forward", "gaining momentum", "exceeding targets", "delivering results"
]

FINANCIAL_RESULTS = [
    "positive returns", "strong performance", "encouraging trends", 
    "favorable projections", "solid fundamentals", "promising indicators",
    "robust growth", "sustainable profits", "market-leading margins"
]

ACHIEVEMENT_ADJECTIVES = [
    "exceptional", "remarkable", "outstanding", "impressive", "strong", "excellent",
    "transformative", "unprecedented", "breakthrough", "industry-defining"
]

SOFTENING_PHRASES = [
    "when you get a chance", "once you've had time to", "if you need any support",
    "at your earliest convenience", "when you have a moment", "if that works for you",
    "when it's convenient", "as your schedule allows", "when you're available",
    "at your discretion", "if you're comfortable with", "pending your approval"
]

TIME_EXPRESSIONS = [
    "next week", "early next week", "later this week", "tomorrow", "this afternoon",
    "after the meeting", "once we've wrapped up", "by end of week", "next quarter",
    "in the coming days", "by month-end", "following our review", "post-implementation"
]

STAKEHOLDER_GROUPS = [
    "the team", "stakeholders", "leadership", "the client", "our partners",
    "the group", "everyone", "the department", "management", "the committee",
    "the board", "investors", "customers", "employees", "the organization"
]

DEPARTMENTS = [
    "Business", "Strategic", "Digital", "Product", "Operations", 
    "Customer", "Technical", "Innovation", "Growth", "Solutions",
    "Executive", "Marketing", "Sales", "Engineering", "Analytics"
]

# ============================================================================
# ENHANCED TEMPLATE GENERATOR
# ============================================================================

def generate_from_template(template, **kwargs):
    """Enhanced template generator supporting all 3 layers"""
    filled_template = template.format(
        # Layer 1: Core corporate vocabulary
        action_verb=random.choice(ACTION_VERBS),
        corporate_verb=random.choice(CORPORATE_VERBS),
        corporate_noun=random.choice(CORPORATE_NOUNS),
        gerund=random.choice(CORPORATE_GERUNDS),
        softening_phrase=random.choice(SOFTENING_PHRASES),
        time_expression=random.choice(TIME_EXPRESSIONS),
        stakeholder_group=random.choice(STAKEHOLDER_GROUPS),
        department=random.choice(DEPARTMENTS),
        performance_adjective=random.choice(PERFORMANCE_ADJECTIVES),
        business_outcome=random.choice(BUSINESS_OUTCOMES),
        project_status=random.choice(PROJECT_STATUS),
        financial_result=random.choice(FINANCIAL_RESULTS),
        achievement_adjective=random.choice(ACHIEVEMENT_ADJECTIVES),
        
        # Layer 2: Specialized scenarios
        interview_skills=random.choice(INTERVIEW_SKILLS),
        interview_qualities=random.choice(INTERVIEW_QUALITIES),
        product_features=random.choice(PRODUCT_FEATURES),
        product_benefits=random.choice(PRODUCT_BENEFITS),
        timeline_phases=random.choice(TIMELINE_PHASES),
        marketing_benefits=random.choice(MARKETING_BENEFITS),
        marketing_features=random.choice(MARKETING_FEATURES),
        
        # Layer 3: Meta-corporate
        vision_concepts=random.choice(VISION_CONCEPTS),
        culture_values=random.choice(CULTURE_VALUES),
        thought_leadership_topics=random.choice(THOUGHT_LEADERSHIP_TOPICS),
        executive_concerns=random.choice(EXECUTIVE_CONCERNS),
        board_metrics=random.choice(BOARD_METRICS),
        quarter=random.choice([1, 2, 3, 4]),
        
        **kwargs  # Custom parameters
    )
    return filled_template

# ============================================================================
# COMPREHENSIVE TRAINING DATA GENERATOR
# ============================================================================

def generate_comprehensive_training_data():
    """Generate training data across all 3 layers"""
    training_data = []
    
    # ========================================================================
    # LAYER 1: Enhanced basic scenarios (from previous version)
    # ========================================================================
    
    # Basic email transformations
    casual_emails = [
        "Hey, can you send me the report?", "Did you finish the project?", 
        "What do you think?", "Can we meet tomorrow?", "I need help with this",
        "Are you available?", "Thanks for your help", "Let me know when it's done",
        "Can you review this?", "I have some feedback"
    ]
    
    basic_email_templates = [
        "Could you {action_verb} {softening_phrase} to {corporate_verb} the {corporate_noun}?",
        "I'll {action_verb} {time_expression} with an update on the {corporate_noun}",
        "Let's {action_verb} on this to ensure we're {gerund} our {corporate_noun} effectively"
    ]
    
    for email in casual_emails:
        for _ in range(2):
            corporate_version = generate_from_template(random.choice(basic_email_templates))
            training_data.append({
                "instruction": "Transform this to professional corporate communication",
                "input": email,
                "output": corporate_version
            })
    
    # ========================================================================
    # LAYER 2: Specialized scenario generation
    # ========================================================================
    
    # Interview scenarios
    interview_inputs = [
        "tell me about yourself", "why do you want this job?", "what are your strengths?",
        "describe a challenge you faced", "where do you see yourself in 5 years?"
    ]
    
    for input_text in interview_inputs:
        # Candidate responses
        for _ in range(2):
            candidate_response = generate_from_template(random.choice(INTERVIEW_TEMPLATES["candidate_response"]))
            training_data.append({
                "instruction": "Respond to this interview question professionally",
                "input": input_text,
                "output": candidate_response
            })
        
        # Interviewer follow-ups
        interviewer_followup = generate_from_template(random.choice(INTERVIEW_TEMPLATES["interviewer_opening"]))
        training_data.append({
            "instruction": "Ask a professional interview question",
            "input": f"interviewer response to: {input_text}",
            "output": interviewer_followup
        })
    
    # Product roadmap communications
    roadmap_inputs = [
        "new feature announcement", "product update", "roadmap priorities", 
        "development timeline", "feature request response"
    ]
    
    for input_text in roadmap_inputs:
        for category in ROADMAP_TEMPLATES:
            for _ in range(2):
                roadmap_output = generate_from_template(random.choice(ROADMAP_TEMPLATES[category]))
                training_data.append({
                    "instruction": f"Create {category.replace('_', ' ')} content",
                    "input": input_text,
                    "output": roadmap_output
                })
    
    # Website and marketing copy
    website_inputs = [
        "describe our product", "why choose us?", "contact us", "get started", 
        "learn more", "product benefits", "company overview"
    ]
    
    for input_text in website_inputs:
        for category in WEBSITE_TEMPLATES:
            website_output = generate_from_template(random.choice(WEBSITE_TEMPLATES[category]))
            training_data.append({
                "instruction": f"Create {category.replace('_', ' ')} website content",
                "input": input_text,
                "output": website_output
            })
    
    # ========================================================================
    # LAYER 3: Meta-corporate content generation
    # ========================================================================
    
    # Vision and culture content
    culture_inputs = [
        "company mission", "our values", "company culture", "what we stand for",
        "our purpose", "vision statement"
    ]
    
    for input_text in culture_inputs:
        for category in VISION_TEMPLATES:
            for _ in range(2):
                vision_output = generate_from_template(random.choice(VISION_TEMPLATES[category]))
                training_data.append({
                    "instruction": f"Create {category.replace('_', ' ')} content",
                    "input": input_text,
                    "output": vision_output
                })
    
    # Executive and board communications
    executive_inputs = [
        "quarterly update", "board presentation", "investor call", "annual report",
        "strategic review", "performance summary"
    ]
    
    for input_text in executive_inputs:
        for category in EXECUTIVE_TEMPLATES:
            executive_output = generate_from_template(random.choice(EXECUTIVE_TEMPLATES[category]))
            training_data.append({
                "instruction": f"Create {category.replace('_', ' ')} content",
                "input": input_text,
                "output": executive_output
            })
    
    # Thought leadership content
    thought_leadership_inputs = [
        "industry trends", "future predictions", "expert opinion", "market analysis",
        "best practices", "innovation insights"
    ]
    
    for input_text in thought_leadership_inputs:
        for _ in range(3):
            thought_output = generate_from_template(random.choice(VISION_TEMPLATES["thought_leadership"]))
            training_data.append({
                "instruction": "Create thought leadership content",
                "input": input_text,
                "output": thought_output
            })
    
    # ========================================================================
    # COMPOUND SCENARIOS (Using multiple layers together)
    # ========================================================================
    
    # Multi-part communications (combining templates)
    compound_scenarios = [
        ("product launch announcement", ["feature_announcement", "timeline_communication", "call_to_action"]),
        ("executive all-hands", ["culture_description", "board_presentation", "thought_leadership"]),
        ("client proposal", ["value_proposition", "feature_description", "interview_followup"]),
        ("annual report section", ["board_presentation", "vision_statement", "achievement_celebration"])
    ]
    
    for scenario_name, template_types in compound_scenarios:
        for _ in range(5):
            # Generate compound output by combining multiple templates
            compound_parts = []
            for template_type in template_types:
                # Find template in appropriate category
                for category_dict in [ROADMAP_TEMPLATES, WEBSITE_TEMPLATES, VISION_TEMPLATES, EXECUTIVE_TEMPLATES]:
                    if template_type in category_dict:
                        part = generate_from_template(random.choice(category_dict[template_type]))
                        compound_parts.append(part)
                        break
            
            compound_output = " ".join(compound_parts)
            training_data.append({
                "instruction": f"Create comprehensive {scenario_name} content",
                "input": scenario_name,
                "output": compound_output
            })
    
    return training_data

def save_comprehensive_dataset(output_path="data/comprehensive_corporate_training.jsonl", num_examples=5000):
    """Save comprehensive 3-layer training dataset"""
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_training_data = []
    
    # Generate multiple batches to reach target number
    while len(all_training_data) < num_examples:
        batch = generate_comprehensive_training_data()
        all_training_data.extend(batch)
    
    # Trim to exact number and shuffle
    training_data = random.sample(all_training_data, num_examples)
    
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(training_data)} comprehensive training examples")
    print("\n🏢 CORPORATE COMMUNICATION COVERAGE:")
    print("=" * 60)
    print("📧 LAYER 1 - Foundational:")
    print("   • Email communications & follow-ups")
    print("   • Meeting requests & scheduling") 
    print("   • Corporate job titles & roles")
    print("   • Project status updates")
    print("   • Performance feedback & reviews")
    print("   • Budget & financial discussions")
    print("\n🎯 LAYER 2 - Specialized:")
    print("   • Job interviews (both sides)")
    print("   • Product roadmaps & announcements")
    print("   • Website copy & marketing materials")
    print("   • Sales presentations & client communication")
    print("   • Feature descriptions & timelines")
    print("   • Value propositions & calls-to-action")
    print("\n🌟 LAYER 3 - Meta-Corporate:")
    print("   • Vision & mission statements")
    print("   • Company culture descriptions")
    print("   • Thought leadership content")
    print("   • Board presentations & investor relations")
    print("   • Annual reports & executive communications")
    print("   • Industry analysis & expert opinions")
    print("\n🔄 COMPOUND SCENARIOS:")
    print("   • Multi-part product launches")
    print("   • Executive all-hands presentations")
    print("   • Comprehensive client proposals")
    print("   • Integrated annual report sections")
    
    # Show sample outputs from each layer
    samples = random.sample(training_data, 6)
    print(f"\n📝 SAMPLE OUTPUTS:")
    print("=" * 60)
    for i, sample in enumerate(samples[:6], 1):
        print(f"{i}. '{sample['input']}' →")
        print(f"   '{sample['output'][:100]}{'...' if len(sample['output']) > 100 else ''}'")
        print()
    
    return output_path

if __name__ == "__main__":
    save_comprehensive_dataset()