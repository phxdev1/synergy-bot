"""CLI interface for SynergyBot"""

import click
from synergy_bot.inference import SynergyBot, MeetingComplicator, TitleInflator
from synergy_bot.train_model import CorporateSpeakTrainer
import os

@click.group()
def cli():
    """SynergyBot - Transform simple language into corporate hellscape"""
    pass

@cli.command()
@click.option('--model-name', default='microsoft/Phi-4', help='Base model to fine-tune')
@click.option('--output-dir', default='models/synergy-bot', help='Output directory for model')
def train(model_name, output_dir):
    """Train the SynergyBot model"""
    click.echo("🏢 Initiating strategic model optimization journey...")
    
    # Generate dataset
    from synergy_bot.dataset_generator import save_dataset
    data_path = save_dataset()
    click.echo(f"✅ Synergized training data at {data_path}")
    
    # Train model
    trainer = CorporateSpeakTrainer(model_name=model_name, output_dir=output_dir)
    trainer.prepare_model()
    trainer.prepare_dataset(data_path)
    trainer.train()
    
    click.echo("🎯 Model transformation complete! Ready to revolutionize communication!")

@cli.command()
@click.argument('text')
@click.option('--model-path', default='models/synergy-bot', help='Path to fine-tuned model')
def synergize(text, model_path):
    """Convert simple text to corporate speak"""
    if not os.path.exists(model_path):
        click.echo("❌ Model not found. Run 'synergy-bot train' first!")
        return
        
    bot = SynergyBot(model_path)
    result = bot.generate_corporate_speak(text)
    click.echo(f"\n📊 Synergized Output:\n{result}")

@cli.command()
@click.argument('meeting')
@click.option('--model-path', default='models/synergy-bot', help='Path to fine-tuned model')
def complicate(meeting, model_path):
    """Complicate a simple meeting request"""
    if not os.path.exists(model_path):
        click.echo("❌ Model not found. Run 'synergy-bot train' first!")
        return
        
    bot = SynergyBot(model_path)
    complicator = MeetingComplicator(bot)
    result = complicator.complicate(meeting)
    click.echo(f"\n📅 Meeting Complication Matrix:\n{result}")

@cli.command()
@click.argument('title')
@click.option('--model-path', default='models/synergy-bot', help='Path to fine-tuned model')
def inflate(title, model_path):
    """Generate inflated corporate title"""
    if not os.path.exists(model_path):
        click.echo("❌ Model not found. Run 'synergy-bot train' first!")
        return
        
    bot = SynergyBot(model_path)
    inflator = TitleInflator(bot)
    result = inflator.inflate(title)
    click.echo(f"\n🏆 Executive Title Enhancement:\n{result}")

@cli.command()
@click.option('--topic', default='business strategy', help='Topic for buzzword generation')
@click.option('--count', default=5, help='Number of buzzword sentences to generate')
@click.option('--model-path', default='models/synergy-bot', help='Path to fine-tuned model')
def buzzwords(topic, count, model_path):
    """Generate corporate buzzword salad"""
    if not os.path.exists(model_path):
        click.echo("❌ Model not found. Run 'synergy-bot train' first!")
        return
        
    bot = SynergyBot(model_path)
    click.echo(f"\n💼 Buzzword Synergy Generator - Topic: {topic}\n")
    
    for i in range(count):
        result = bot.buzzword_generator(topic)
        click.echo(f"{i+1}. {result}")

@cli.command()
@click.option('--model-path', default='models/synergy-bot', help='Path to fine-tuned model')
def interactive(model_path):
    """Interactive corporate speak transformer"""
    if not os.path.exists(model_path):
        click.echo("❌ Model not found. Run 'synergy-bot train' first!")
        return
        
    bot = SynergyBot(model_path)
    complicator = MeetingComplicator(bot)
    inflator = TitleInflator(bot)
    
    click.echo("🏢 Welcome to SynergyBot Interactive Mode!")
    click.echo("Commands: 'synergize', 'meeting', 'title', 'buzzword', 'quit'\n")
    
    while True:
        command = click.prompt("Enter command", type=click.Choice(['synergize', 'meeting', 'title', 'buzzword', 'quit']))
        
        if command == 'quit':
            click.echo("Cascading session termination across all stakeholders... Goodbye! 👋")
            break
            
        elif command == 'synergize':
            text = click.prompt("Enter text to synergize")
            result = bot.generate_corporate_speak(text)
            click.echo(f"\n📊 {result}\n")
            
        elif command == 'meeting':
            meeting = click.prompt("Enter meeting request")
            result = complicator.complicate(meeting)
            click.echo(f"\n📅 {result}\n")
            
        elif command == 'title':
            title = click.prompt("Enter job title")
            result = inflator.inflate(title)
            click.echo(f"\n🏆 {result}\n")
            
        elif command == 'buzzword':
            topic = click.prompt("Enter topic", default="business strategy")
            result = bot.buzzword_generator(topic)
            click.echo(f"\n💼 {result}\n")

def main():
    cli()

if __name__ == '__main__':
    main()