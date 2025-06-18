from setuptools import setup, find_packages

setup(
    name="synergy-bot",
    version="0.1.0",
    description="Corporate jargon and synergy generator bot",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "accelerate>=0.21.0",
        "peft>=0.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "flask>=3.0.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "synergy-bot=synergy_bot.cli:main",
        ],
    },
)