#!/bin/bash

echo "🚀 Pushing SynergyBot to Replicate"
echo "================================="
echo ""
echo "Step 1: Make sure you're logged in to Replicate"
echo "If not, run: ./cog login"
echo ""
echo "Step 2: Pushing model to r8.im/phxdev1/synergy-bot"
echo ""

# Check if cog build was successful
if [ ! -f ".cog/tmp/build.json" ]; then
    echo "❌ Build not found. Running cog build first..."
    ./cog build
fi

# Push to Replicate
echo "📤 Pushing to Replicate..."
./cog push r8.im/phxdev1/synergy-bot

echo ""
echo "✅ Push complete!"
echo ""
echo "Your model will be available at:"
echo "🌐 https://replicate.com/phxdev1/synergy-bot"
echo ""
echo "Test with:"
echo "./cog predict r8.im/phxdev1/synergy-bot:latest -i text='let us meet' -i mode='complicate_meeting'"