#!/bin/bash
# =============================================================================
# SIMPLIFIED TRAINING SCRIPT FOR NANOGPT WITH ZERO-ORDER OPTIMIZATION
# =============================================================================
# This script runs training using the comprehensive config files.
# Usage: bash lozo_train.sh [config_type] [single|multi]
#
# Config types:
#   - first-order: Adam, SGD (train_first_order_config.py)
#   - mezo: MeZO, MeZO-M, DiMeZO (train_mezo_config.py)
#   - lozo: LoZO, LoZO-M, SVD-LoZO, DiLoZO (train_lozo_config.py)
#   - kronzo: KronZO, DiKronZO (train_kronzo_config.py)

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
CONFIG_TYPE=${1:-lozo}    # Config type: first-order, mezo, lozo, kronzo
MODE=${2:-single}         # Execution mode: single or multi (for multi-GPU)

# Validate config type
case "$CONFIG_TYPE" in
    first-order|mezo|lozo|kronzo)
        ;;
    *)
        echo "Error: Invalid config type '$CONFIG_TYPE'"
        echo "Valid options: first-order, mezo, lozo, kronzo"
        echo ""
        echo "Usage: bash lozo_train.sh [config_type] [single|multi]"
        echo ""
        echo "Examples:"
        echo "  bash lozo_train.sh lozo single          # LoZO on single GPU"
        echo "  bash lozo_train.sh mezo multi           # MeZO on multiple GPUs"
        echo "  bash lozo_train.sh first-order single   # Adam/SGD on single GPU"
        echo "  bash lozo_train.sh kronzo single        # KronZO on single GPU"
        exit 1
        ;;
esac

# Validate execution mode
case "$MODE" in
    single|multi)
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Valid options: single, multi"
        exit 1
        ;;
esac

# =============================================================================
# CONFIG FILE SELECTION
# =============================================================================
case "$CONFIG_TYPE" in
    first-order)
        CONFIG="config/train_first_order_config.py"
        SCRIPT="train.py"  # Use standard training script for first-order methods
        ;;
    mezo)
        CONFIG="config/train_mezo_config.py"
        SCRIPT="lozo_train.py"  # Use zero-order training script
        ;;
    lozo)
        CONFIG="config/train_lozo_config.py"
        SCRIPT="lozo_train.py"  # Use zero-order training script
        ;;
    kronzo)
        CONFIG="config/train_kronzo_config.py"
        SCRIPT="lozo_train.py"  # Use zero-order training script
        ;;
esac

# =============================================================================
# INTELLIGENT LOGGING SETUP
# =============================================================================
# Generate intelligent log name based on config parameters
echo "Analyzing configuration for intelligent log naming..."
INTELLIGENT_LOG_NAME=$(python generate_log_name.py "$CONFIG")

if [ $? -ne 0 ] || [ -z "$INTELLIGENT_LOG_NAME" ]; then
    echo "Warning: Could not generate intelligent log name, using fallback"
    INTELLIGENT_LOG_NAME="${CONFIG_TYPE}-fallback"
fi

# Create timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Create log directory with method-dataset structure
LOG_DIR="logs/${INTELLIGENT_LOG_NAME}"
mkdir -p "$LOG_DIR"

# Create descriptive log file name
LOG_FILE="${LOG_DIR}/${INTELLIGENT_LOG_NAME}-${TIMESTAMP}.log"

# =============================================================================
# TRAINING EXECUTION
# =============================================================================
echo "============================================================================="
echo "NANOGPT TRAINING WITH ZERO-ORDER OPTIMIZATION"
echo "============================================================================="
echo "Config type: $CONFIG_TYPE"
echo "Config file: $CONFIG"
echo "Training script: $SCRIPT"
echo "Execution mode: $MODE"
echo ""
echo "📁 Log directory: $LOG_DIR"
echo "📄 Log file: $LOG_FILE"
echo ""
echo "Log naming breakdown:"
echo "  - Intelligent name: $INTELLIGENT_LOG_NAME"
echo "  - Timestamp: $TIMESTAMP"
echo "============================================================================="

if [ "$MODE" = "single" ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    python $SCRIPT $CONFIG 2>&1 | tee "$LOG_FILE"
    
elif [ "$MODE" = "multi" ]; then
    # Multi-GPU training with DDP
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Starting multi-GPU training on $NUM_GPUS GPUs..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS $SCRIPT $CONFIG 2>&1 | tee "$LOG_FILE"
    
else
    echo "Error: Invalid mode '$MODE'"
    exit 1
fi

echo "============================================================================="
echo "Training completed. Log saved to: $LOG_FILE"
echo "=============================================================================" 