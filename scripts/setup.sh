#!/bin/bash
# KB-Compiler Setup Script

set -e

echo "🚀 KB-Compiler Setup"
echo "===================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ required, found $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check for KIMI_API_KEY
if [ -z "$KIMI_API_KEY" ]; then
    echo "⚠️  Warning: KIMI_API_KEY not set"
    echo "   Set it with: export KIMI_API_KEY='your-key'"
else
    echo "✅ KIMI_API_KEY is set"
fi

# Install package
echo ""
echo "📦 Installing KB-Compiler..."
pip install -e "." -q

# Check obsidian-cli
if command -v obsidian-cli &> /dev/null; then
    echo "✅ obsidian-cli found"
else
    echo "⚠️  obsidian-cli not found (optional but recommended)"
    echo "   Install with: brew install yuangziweigithub/tap/obsidian-cli"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Set KIMI_API_KEY: export KIMI_API_KEY='your-key'"
echo "  2. Initialize KB: kb-compiler init ~/KnowledgeBase"
echo "  3. Add documents to ~/KnowledgeBase/raw/"
echo "  4. Compile: kb-compiler compile"
