"""Example usage of SynergyBot with Cog"""

import subprocess
import json

def run_cog_prediction(text, mode="synergize", temperature=0.8):
    """Run a Cog prediction"""
    cmd = [
        "./cog", "predict",
        "-i", f"text={text}",
        "-i", f"mode={mode}",
        "-i", f"temperature={temperature}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# Examples
examples = [
    ("Let's have a quick sync", "synergize"),
    ("Can we meet at 3pm?", "complicate_meeting"),
    ("Software Engineer", "inflate_title"),
    ("Our sales strategy", "buzzword"),
    ("Hi, Please send me the report. Thanks.", "email")
]

print("🏢 SynergyBot Examples\n" + "="*50)

for text, mode in examples:
    print(f"\n📝 Input: '{text}'")
    print(f"🔧 Mode: {mode}")
    print(f"💼 Output: {run_cog_prediction(text, mode)}")
    print("-"*50)