#!/bin/bash

# Deploy SynergyBot to Replicate

echo "🚀 Deploying SynergyBot to Replicate..."

# Build the container
echo "📦 Building Cog container..."
./cog build

# Push to Replicate (requires authentication)
echo "☁️ Pushing to Replicate..."
echo "Make sure you've run: cog login"
echo "Then run: cog push r8.im/phxdev1/synergy-bot"

# Example commands:
# cog login
# cog push r8.im/yourusername/synergy-bot

echo -e "\n📝 After pushing, you can run predictions via:"
echo "cog predict r8.im/phxdev1/synergy-bot:latest -i text='simple meeting' -i mode='complicate_meeting'"