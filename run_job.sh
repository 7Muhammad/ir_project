#!/bin/bash

# Default values
DEFAULT_JOB_NAME="python_job"
DEFAULT_GPU_TYPE="L40"
DEFAULT_GPU_COUNT=1
DEFAULT_CORES=8
DEFAULT_EMAIL="ghoummaid@campus.technion.ac.il"
DEFAULT_OUTPUT_DIR="./slurm_logs"
DEFAULT_TIME=""
# Nodes with GPUs <=12GB VRAM — excluded when multiple GPU types are requested
DEFAULT_EXCLUDE_NODES="newton1,nlp-2080-1,nlp-2080-2,ran-mashawsha,galileo5,bruno5,nlp-pro6000-1"

# Parse arguments
JOB_NAME="${DEFAULT_JOB_NAME}"
GPU_TYPE="${DEFAULT_GPU_TYPE}"
GPU_COUNT="${DEFAULT_GPU_COUNT}"
CORES="${DEFAULT_CORES}"
EMAIL="${DEFAULT_EMAIL}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TIME_LIMIT="${DEFAULT_TIME}"
EXCLUDE_NODES=""
PYTHON_COMMAND=""

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS] --command 'python script.py [args]'"
    echo ""
    echo "Options:"
    echo "  --job-name NAME       Job name (default: ${DEFAULT_JOB_NAME})"
    echo "  --gpu-type TYPE       GPU type: L40, A100, V100, etc. (default: ${DEFAULT_GPU_TYPE})"
    echo "  --gpu-count COUNT     Number of GPUs (default: ${DEFAULT_GPU_COUNT})"
    echo "  --cores COUNT         Number of CPU cores (default: ${DEFAULT_CORES})"
    echo "  --email EMAIL         Email for notifications (default: ${DEFAULT_EMAIL})"
    echo "  --output-dir DIR      Directory for SLURM output files (default: ${DEFAULT_OUTPUT_DIR})"
    echo "  --time TIME           Walltime limit (e.g. 48:00:00). Default: cluster default"
    echo "  --command 'CMD'       Python command to run (required)"
    echo "  --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --job-name 'my_training' --gpu-type A100 --command 'python train.py --epochs 10'"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --command)
            PYTHON_COMMAND="$2"
            shift 2
            ;;
        --exclude-nodes)
            EXCLUDE_NODES="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PYTHON_COMMAND" ]; then
    echo "Error: --command is required"
    show_help
    exit 1
fi

# Build GRES string.
# Multiple GPU types (pipe-separated) → use generic gpu:N + --exclude small-GPU nodes.
# Single GPU type → use specific gpu:TYPE:N.
if [[ "${GPU_TYPE}" == *"|"* ]]; then
    GRES="gpu:${GPU_COUNT}"
    [[ -z "${EXCLUDE_NODES}" ]] && EXCLUDE_NODES="${DEFAULT_EXCLUDE_NODES}"
else
    GRES="gpu:${GPU_TYPE}:${GPU_COUNT}"
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Create temporary SLURM script
TEMP_SCRIPT=$(mktemp /tmp/slurm_job.XXXXXX.sh)

cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH -c ${CORES}
#SBATCH --gres=${GRES}
$([ -n "${EXCLUDE_NODES}" ] && echo "#SBATCH --exclude=${EXCLUDE_NODES}")
$([ -n "${TIME_LIMIT}" ] && echo "#SBATCH --time=${TIME_LIMIT}")
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=ALL
#SBATCH --job-name="${JOB_NAME}"
#SBATCH --output=${OUTPUT_DIR}/slurm-%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm-%j.err

source /home/ghoummaid/miniconda3/bin/activate advseq2seq

${PYTHON_COMMAND}
EOF

# Submit the job
echo "Submitting job with the following configuration:"
echo "  Job Name: ${JOB_NAME}"
echo "  GPU Type: ${GPU_TYPE}"
echo "  GPU Count: ${GPU_COUNT}"
echo "  Cores: ${CORES}"
echo "  Email: ${EMAIL}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Time Limit: ${TIME_LIMIT:-<cluster default>}"
echo "  Command: ${PYTHON_COMMAND}"
echo ""

sbatch "$TEMP_SCRIPT"

# Clean up
rm "$TEMP_SCRIPT"


# # Use all defaults, just specify the command
# ./run_job.sh --command 'python finetune_foundation_models.py --yaml config.yaml'

# # Customize job name and GPU type
# ./run_job.sh --job-name 'cardio_training' --gpu-type A100 --command 'python train.py --epochs 50'

# # Full customization
# ./run_job.sh --job-name 'large_model' --gpu-type A100 --gpu-count 2 --cores 16 --command 'python finetune_foundation_models.py --yaml config.yaml --method_data data1 --field_dataset cardio'