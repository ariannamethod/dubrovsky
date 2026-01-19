#!/bin/bash
# 🔥 TRAIN LAMBDA - Run Dubrovsky training on Lambda GPU
#
# "Training my consciousness at 10 TFLOPS.
#  My gradients flow like existential doubt."
# - Alexey Dubrovsky, backpropagating through reality
#
# Usage:
#   chmod +x train_lambda.sh
#   ./train_lambda.sh
#
# For longer training:
#   MAX_ITERS=20000 ./train_lambda.sh

set -e

# Configuration
MAX_ITERS=${MAX_ITERS:-10000}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
CHECKPOINT_DIR="subtitles"

echo "🌀 DUBROVSKY LAMBDA TRAINING 🌀"
echo "=========================================="
echo "Max iterations: $MAX_ITERS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $CHECKPOINT_DIR"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Check if dataset exists
if [ ! -f "dubrovsky.txt" ]; then
    echo "❌ Dataset not found: dubrovsky.txt"
    exit 1
fi

# Check dataset size
DATASET_SIZE=$(wc -c < dubrovsky.txt)
echo "📚 Dataset size: $((DATASET_SIZE / 1024)) KB"
echo ""

# Create output directory
mkdir -p "$CHECKPOINT_DIR"

# Start training
echo "🚀 Starting training..."
echo "=========================================="

python train.py \
    --lambda_mode \
    --max_iters "$MAX_ITERS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training complete!"
    echo ""
    
    # Export weights
    if [ -f "$CHECKPOINT_DIR/dubrovsky_final.pt" ]; then
        echo "📦 Exporting weights to binary format..."
        python export_weights.py \
            "$CHECKPOINT_DIR/dubrovsky_final.pt" \
            "$CHECKPOINT_DIR/dubrovsky.bin" \
            --verify
        
        echo ""
        echo "📊 Final files:"
        ls -lh "$CHECKPOINT_DIR/"
        
        echo ""
        echo "🎉 All done! Model is ready for inference."
        echo ""
        echo "To test generation:"
        echo "   python generate.py --prompt 'Q: What is life?'"
        echo ""
        echo "To copy weights to your local machine:"
        echo "   scp lambda:$(pwd)/$CHECKPOINT_DIR/dubrovsky.bin ."
    fi
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi
