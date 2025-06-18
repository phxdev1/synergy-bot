#!/bin/bash

echo "🏢 Testing SynergyBot with Cog..."

# Test different modes
echo -e "\n📊 Test 1: Synergize simple text"
./cog predict -i text="let's fix this bug" -i mode="synergize"

echo -e "\n📅 Test 2: Complicate a meeting"
./cog predict -i text="can we meet tomorrow?" -i mode="complicate_meeting"

echo -e "\n🏆 Test 3: Inflate a job title"
./cog predict -i text="developer" -i mode="inflate_title"

echo -e "\n💼 Test 4: Generate buzzwords"
./cog predict -i text="product launch" -i mode="buzzword"

echo -e "\n✉️ Test 5: Transform an email"
./cog predict -i text="Hi team, the project is delayed. We need to work faster. Thanks, John" -i mode="email"

echo -e "\n🔥 Test 6: High temperature (more creative)"
./cog predict -i text="we need more sales" -i temperature=1.5