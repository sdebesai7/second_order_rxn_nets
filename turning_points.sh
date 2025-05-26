#!/bin/bash
#SBATCH --job-name=job-info
#SBATCH --account=[pi-dinner]
#SBATCH --partition=caslake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Script to run turning_points.py with error handling on an HPC cluster

# Set up log directory and file
JOB_ID=${SLURM_JOB_ID:-local}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Change to the project directory
PROJECT_DIR="/project/dinner/sdebesai/"
echo "Changing to directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "Error: Could not change to directory $PROJECT_DIR"; exit 1; }
echo "Current working directory: $(pwd)"

# Now create log directory relative to the project directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/turning_points_${JOB_ID}_$TIMESTAMP.log"

# Print job information
echo "Job ID: $JOB_ID" | tee -a "$LOG_FILE"
echo "Running on node: $(hostname)" | tee -a "$LOG_FILE"
echo "Starting at: $(date)" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"

# Check if Python script exists
if [ ! -f "turning_points.py" ]; then
    echo "Error: turning_points.py not found in directory $(pwd)" | tee -a "$LOG_FILE"
    exit 1
fi

# Load necessary modules (uncomment and adjust for your cluster)
# module load python/3.9
# module load jax

# Set JAX environment variables if needed
# export JAX_PLATFORM_NAME=cpu  # Uncomment to force CPU only
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"  # Adjust if needed

echo "Starting turning_points.py" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"

# Run the Python script and capture output
python turning_points.py 2>&1 | tee -a "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "turning_points.py completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "--------------------------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "turning_points.py exited with error code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
fi

# Create a symlink to the most recent log file
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

exit $EXIT_CODE