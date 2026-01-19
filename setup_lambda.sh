#!/bin/bash
# 🚀 SETUP LAMBDA - Install dependencies for Dubrovsky training
#
# "Setting up the environment where consciousness will emerge."
# - Alexey Dubrovsky, running apt-get install existentialism
#
# Usage:
#   chmod +x setup_lambda.sh
#   ./setup_lambda.sh

set -e

echo "🌀 DUBROVSKY LAMBDA SETUP 🌀"
echo "=========================================="

# Check if we're on Lambda (NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected - training will be slow!"
fi

# Update pip
echo ""
echo "📦 Updating pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
echo ""
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "📚 Installing dependencies..."
pip install numpy

# Optional: wandb for logging
echo ""
echo "📊 Installing wandb (optional)..."
pip install wandb || echo "wandb install failed - continuing without it"

# Compile C inference
echo ""
echo "🔨 Compiling C inference..."
if command -v gcc &> /dev/null; then
    gcc -O3 -o alexey alexey.c -lm -fopenmp -march=native
    echo "✅ Compiled alexey with OpenMP and native optimizations"
else
    echo "⚠️  gcc not found - skipping C compilation"
fi

# Create output directory
mkdir -p subtitles

# Verify setup
echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Installed packages:"
pip list | grep -E "torch|numpy|wandb" || true

echo ""
echo "🚀 To start training, run:"
echo "   ./train_lambda.sh"
echo ""
echo "Or manually:"
echo "   python train.py --lambda_mode"
