"""
Fix tokenizer loading issue for Mistral-7B
"""
from transformers import AutoTokenizer
import os

# Clear cache to ensure fresh download
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Attempting to load tokenizer with different methods...")

# Method 1: Force re-download
try:
    print("\nMethod 1: Force re-download...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        force_download=True,
        resume_download=False
    )
    print("✓ Success with force download!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Method 2: Use specific revision
try:
    print("\nMethod 2: Use specific revision...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision="main",
        use_fast=True
    )
    print("✓ Success with specific revision!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Method 3: Try without fast tokenizer
try:
    print("\nMethod 3: Disable fast tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )
    print("✓ Success without fast tokenizer!")
except Exception as e:
    print(f"✗ Failed: {e}")

# Method 4: Clear cache and retry
try:
    print("\nMethod 4: Clear specific model cache...")
    import shutil
    model_cache = os.path.join(cache_dir, f"models--mistralai--Mistral-7B-Instruct-v0.2")
    if os.path.exists(model_cache):
        shutil.rmtree(model_cache)
        print(f"Cleared cache at {model_cache}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Success after clearing cache!")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nIf all methods fail, try:")
print("1. pip install --upgrade transformers")
print("2. pip install --upgrade tokenizers")
print("3. Use a different model like 'microsoft/phi-2' or 'google/gemma-2b'")