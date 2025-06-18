"""Corporate jargon vocabulary and patterns"""

CORPORATE_VERBS = [
    "leverage", "synergize", "optimize", "streamline", "amplify",
    "actualize", "incentivize", "monetize", "revolutionize", "disrupt",
    "pivot", "iterate", "scale", "evangelize", "socialize",
    "operationalize", "crystallize", "ideate", "orchestrate", "cultivate"
]

CORPORATE_NOUNS = [
    "synergy", "paradigm", "ecosystem", "bandwidth", "mindshare",
    "touchpoint", "deliverable", "takeaway", "learnings", "alignment",
    "visibility", "traction", "runway", "headwind", "tailwind",
    "stakeholder", "thought leader", "change agent", "value proposition",
    "best practice", "core competency", "key performance indicator"
]

CORPORATE_ADJECTIVES = [
    "holistic", "robust", "scalable", "disruptive", "innovative",
    "agile", "lean", "mission-critical", "best-in-class", "world-class",
    "cutting-edge", "bleeding-edge", "next-generation", "game-changing",
    "transformative", "strategic", "tactical", "actionable", "impactful"
]

BUZZWORD_PHRASES = [
    "move the needle", "think outside the box", "low-hanging fruit",
    "win-win situation", "circle back", "deep dive", "boil the ocean",
    "drink the Kool-Aid", "eat your own dog food", "run it up the flagpole",
    "put a pin in it", "take it offline", "level set", "table stakes",
    "north star", "30,000 foot view", "boots on the ground", "blue ocean",
    "paradigm shift", "value add", "core competency", "strategic alignment"
]

MEETING_COMPLICATIONS = [
    "Let's schedule a follow-up to discuss the action items",
    "We need to align all stakeholders before moving forward",
    "I think we need a working group to ideate on this",
    "Let's put together a tiger team to tackle this initiative",
    "We should socialize this with the broader organization",
    "I'd like to see a SWOT analysis before we proceed",
    "Can we get a deck together to present to leadership?",
    "We need to ensure we have buy-in from all verticals",
    "Let's take this offline and circle back next week",
    "We should probably loop in legal and compliance"
]

CORPORATE_TITLES = [
    "Chief {noun} Officer",
    "Senior Vice President of {adjective} {noun}",
    "Director of {noun} Excellence",
    "Head of {adjective} Innovation",
    "{noun} Evangelist",
    "Global {noun} Strategist",
    "{adjective} Solutions Architect",
    "Principal {noun} Engineer",
    "{noun} Transformation Lead",
    "Executive {noun} Consultant"
]

def generate_corporate_title():
    """Generate a meaningless corporate title"""
    import random
    
    template = random.choice(CORPORATE_TITLES)
    title = template.format(
        noun=random.choice(CORPORATE_NOUNS).title(),
        adjective=random.choice(CORPORATE_ADJECTIVES).title()
    )
    return title

def complicate_simple_statement(statement):
    """Transform a simple statement into corporate speak"""
    import random
    
    complications = [
        f"Let's leverage our {random.choice(CORPORATE_ADJECTIVES)} {random.choice(CORPORATE_NOUNS)} to {statement.lower()}",
        f"We need to {random.choice(CORPORATE_VERBS)} our approach to {statement.lower()}",
        f"I think we should {random.choice(BUZZWORD_PHRASES)} when it comes to {statement.lower()}",
        f"From a {random.choice(CORPORATE_ADJECTIVES)} perspective, {statement.lower()} requires alignment",
        f"To {statement.lower()}, we'll need to {random.choice(CORPORATE_VERBS)} our {random.choice(CORPORATE_NOUNS)}"
    ]
    
    return random.choice(complications)