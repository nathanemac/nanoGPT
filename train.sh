#!/bin/bash

# Script to run nanoGPT training with method-specific configurations and intelligent logging.

# Default values
DEFAULT_ARGS="" # e.g., --compile=False --eval_interval=100

# Detect if we're running from inside nanoGPT directory or from parent
if [ -f "zo_train.py" ]; then
    # Running from inside nanoGPT directory
    TRAIN_SCRIPT="zo_train.py"
    CONFIG_DIR="config"
    LOG_NAME_GENERATOR="generate_log_name.py"
else
    # Running from parent directory
    TRAIN_SCRIPT="nanoGPT/zo_train.py"
    CONFIG_DIR="nanoGPT/config"
    LOG_NAME_GENERATOR="nanoGPT/generate_log_name.py"
fi

# Create logs directory if it doesn't exist
LOGS_DIR="logs"
mkdir -p "${LOGS_DIR}"

# --- Method to Config File Mapping ---
declare -A METHOD_CONFIG_MAP
METHOD_CONFIG_MAP["lozo"]="train_lozo_config.py"
METHOD_CONFIG_MAP["mezo"]="train_mezo_config.py"
METHOD_CONFIG_MAP["kronzo"]="train_kronzo_config.py"
METHOD_CONFIG_MAP["adam"]="train_first_order_config.py"
METHOD_CONFIG_MAP["sgd"]="train_first_order_config.py"

# --- Helper Function ---
print_usage() {
    echo "Usage: $0 <train_method> [gpu_type] [additional_python_args...]"
    echo ""
    echo "Arguments:"
    echo "  <train_method>         : The training method category to use."
    echo "                           This will determine which config file is loaded."
    echo "  [gpu_type]             : Optional. Type of GPU setup. Can be 'single' or 'ddp'."
    echo "                           'single' (default): Runs on a single GPU."
    echo "                           'ddp': Runs with Distributed Data Parallel using torchrun."
    echo "                                  Additional DDP args like --nproc_per_node can be passed after this."
    echo "  [additional_python_args...] : Any other arguments to pass directly to the ${TRAIN_SCRIPT}."
    echo "                                  (e.g., --batch_size=32 --learning_rate=1e-3)"
    echo ""
    echo "Example (single GPU):"
    echo "  $0 lozo single --batch_size=64 --compile=False"
    echo "  $0 adam single --dataset=shakespeare_char"
    echo ""
    echo "Example (DDP with torchrun, 4 GPUs):"
    echo "  $0 mezo ddp --nproc_per_node=4 --batch_size=32"
    echo ""
    echo "Available training method categories:"
    echo "  - lozo     : LoZO-based methods (uses train_lozo_config.py)"
    echo "               Config specifies: lozo, lozom, svdlozo, dilozo variants"
    echo "  - mezo     : MeZO-based methods (uses train_mezo_config.py)"
    echo "               Config specifies: mezo, mezom, dimezo variants"
    echo "  - kronzo   : KronZO-based methods (uses train_kronzo_config.py)"
    echo "               Config specifies: kronzo, dikronzo, improved_kronzo variants"
    echo "               Set train_method='improved_kronzo' in config for advanced version"
    echo "  - adam     : First-order methods (uses train_first_order_config.py)"
    echo "               Config specifies: adam optimizer"
    echo "  - sgd      : First-order methods (uses train_first_order_config.py)"
    echo "               Config specifies: sgd optimizer"
    echo ""
    echo "Note: The exact method variant (e.g., lozom vs dilozo) is specified"
    echo "      in the config file's 'train_method' parameter."
    echo ""
    echo "Logs will be saved to: ${LOGS_DIR}/<intelligent_log_name>.log"
    echo ""
}

# --- Argument Parsing ---
if [ "$#" -lt 1 ]; then
    echo "Error: No training method specified."
    print_usage
    exit 1
fi

TRAIN_METHOD=$1
shift # Remove train_method from arguments

# Check if the method is supported
if [[ ! -v METHOD_CONFIG_MAP["$TRAIN_METHOD"] ]]; then
    echo "Error: Unsupported training method '${TRAIN_METHOD}'"
    echo "Supported methods: ${!METHOD_CONFIG_MAP[@]}"
    print_usage
    exit 1
fi

# Get the config file for this method
CONFIG_FILE="${METHOD_CONFIG_MAP[$TRAIN_METHOD]}"
METHOD_CONFIG_FILE="${CONFIG_DIR}/${CONFIG_FILE}"

if [ ! -f "$METHOD_CONFIG_FILE" ]; then
    echo "Error: Configuration file '${CONFIG_FILE}' not found at ${METHOD_CONFIG_FILE}"
    echo "Make sure the config file exists in the ${CONFIG_DIR} directory."
    echo "Current working directory: $(pwd)"
    echo "Looking for: ${METHOD_CONFIG_FILE}"
    exit 1
fi

# Generate intelligent log name using the config file
if [ -f "$LOG_NAME_GENERATOR" ]; then
    LOG_NAME=$(python "$LOG_NAME_GENERATOR" "$METHOD_CONFIG_FILE" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$LOG_NAME" ]; then
        echo "Warning: Failed to generate intelligent log name, using fallback"
        LOG_NAME="${TRAIN_METHOD}-$(date +%Y%m%d_%H%M%S)"
    fi
else
    echo "Warning: Log name generator not found at $LOG_NAME_GENERATOR, using fallback"
    LOG_NAME="${TRAIN_METHOD}-$(date +%Y%m%d_%H%M%S)"
fi

# Add timestamp to make log name unique
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/${LOG_NAME}_${TIMESTAMP}.log"

GPU_TYPE="single" # Default to single GPU
if [[ "$1" == "single" || "$1" == "ddp" ]]; then
    GPU_TYPE=$1
    shift # Remove gpu_type from arguments
fi

# Remaining arguments are passed to the python script
PYTHON_ARGS="$@"

# Construct the command - let the config file specify the exact train_method
CMD_ARGS="--method_config_file ${METHOD_CONFIG_FILE} ${DEFAULT_ARGS} ${PYTHON_ARGS}"

echo "--------------------------------------------------"
echo "Starting nanoGPT training with configuration:"
echo "  Method Category: ${TRAIN_METHOD}"
echo "  Config File    : ${CONFIG_FILE}"
echo "  Config Path    : ${METHOD_CONFIG_FILE}"
echo "  GPU Setup      : ${GPU_TYPE}"
echo "  Python Script  : ${TRAIN_SCRIPT}"
echo "  Script Args    : ${CMD_ARGS}"
echo "  Log File       : ${LOG_FILE}"
echo "--------------------------------------------------"
echo ""
echo "📝 Training output will be logged to:"
echo "   ${LOG_FILE}"
echo ""
echo "💡 To monitor progress in real-time, run:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "--------------------------------------------------"

# Function to execute command with logging
execute_with_logging() {
    local cmd="$1"
    echo "Executing: $cmd" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    
    # Execute command and capture both stdout and stderr to log file
    # while also displaying output in real-time
    eval "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Finished at: $(date)" | tee -a "$LOG_FILE"
    echo "Exit code: $exit_code" | tee -a "$LOG_FILE"
    
    return $exit_code
}

# Execute based on GPU type
if [ "${GPU_TYPE}" == "ddp" ]; then
    # For DDP, nproc_per_node and other torchrun args should be part of PYTHON_ARGS
    # Example: $0 lozo ddp --nproc_per_node=4 --other_script_arg=value
    # The user must provide --nproc_per_node (or other relevant torchrun flags) in additional_python_args
    # We will extract common torchrun args if present, otherwise user must ensure they are there.
    
    NPROC_PER_NODE_ARG=""
    TORCHRUN_SPECIFIC_ARGS="" # For standalone, nnodes, etc.

    # Basic extraction for nproc_per_node for convenience
    # More sophisticated parsing could be added if needed
    TEMP_PYTHON_ARGS=""
    FOUND_NPROC=false
    for arg in $PYTHON_ARGS; do
        if [[ "$arg" == --nproc_per_node=* ]]; then
            NPROC_PER_NODE_ARG="$arg"
            FOUND_NPROC=true
        elif [[ "$arg" == "--nproc_per_node" ]]; then
            # Handle space separated --nproc_per_node value
            NPROC_PER_NODE_ARG="$arg"
            FOUND_NPROC=true # Next arg will be the value
        elif $FOUND_NPROC && [[ ! "$NPROC_PER_NODE_ARG" == --nproc_per_node=* ]] ; then
             NPROC_PER_NODE_ARG="$NPROC_PER_NODE_ARG $arg"
             FOUND_NPROC="processed" 
        else
            TEMP_PYTHON_ARGS="${TEMP_PYTHON_ARGS} ${arg}"
        fi
    done
    PYTHON_ARGS=$(echo "$TEMP_PYTHON_ARGS" | xargs) # Remove extra spaces
    CMD_ARGS="--method_config_file ${METHOD_CONFIG_FILE} ${DEFAULT_ARGS} ${PYTHON_ARGS}"

    if [[ -z "$NPROC_PER_NODE_ARG" ]] && [[ ! " ${PYTHON_ARGS} " =~ " --nproc_per_node " ]]; then
        echo "Warning: Running in DDP mode, but --nproc_per_node is not specified." | tee -a "$LOG_FILE"
        echo "         torchrun might default or fail. Consider adding it, e.g., train.sh <method> ddp --nproc_per_node=4" | tee -a "$LOG_FILE"
    fi
    
    # Execute with torchrun and logging
    FULL_CMD="torchrun ${NPROC_PER_NODE_ARG} ${TORCHRUN_SPECIFIC_ARGS} ${TRAIN_SCRIPT} ${CMD_ARGS}"
    execute_with_logging "$FULL_CMD"
    TRAINING_EXIT_CODE=$?
else
    # Single GPU with logging
    FULL_CMD="python ${TRAIN_SCRIPT} ${CMD_ARGS}"
    execute_with_logging "$FULL_CMD"
    TRAINING_EXIT_CODE=$?
fi

echo "--------------------------------------------------" | tee -a "$LOG_FILE"
echo "Training finished for method: ${TRAIN_METHOD}" | tee -a "$LOG_FILE"
echo "Final exit code: ${TRAINING_EXIT_CODE}" | tee -a "$LOG_FILE"
echo "--------------------------------------------------" | tee -a "$LOG_FILE"

echo ""
echo "📊 Training completed! Log saved to:"
echo "   ${LOG_FILE}"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: ${TRAINING_EXIT_CODE}"
    echo "   Check the log file for details: ${LOG_FILE}"
fi

exit $TRAINING_EXIT_CODE 