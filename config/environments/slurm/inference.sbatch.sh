#!/bin/bash
#SBATCH --job-name=antisemitism_detection
#SBATCH --output=antisemitism_detection_%j.out
#SBATCH --error=antisemitism_detection_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=ai

# Load necessary modules (customize for your cluster)
module load python/3.9
module load cuda/11.7

# Set environment variables
export PROJECT_ROOT="${HOME}/antisemitism_detection"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Change to project directory
cd ${PROJECT_ROOT}

# Activate virtual environment if needed
source ${HOME}/venvs/antisemitism/bin/activate

# Set up port forwarding for Ollama if using remote instance
# SSH into the Ollama server and forward the port
SSH_PORT_FWD="ssh -N -L 11434:localhost:11434 ollama_server &"
eval ${SSH_PORT_FWD}
SSH_PID=$!

# Wait for port forwarding to establish
sleep 5

# Run the inference script
python scripts/run_inference.py \
    --input data/raw/tweets.csv \
    --output data/results/analysis_$(date +%Y%m%d_%H%M%S).csv \
    --config ollama_config \
    --definitions IHRA,JDA \
    --max-workers 16

# Kill the SSH port forwarding process
if [ ! -z "$SSH_PID" ]; then
    kill $SSH_PID
fi

# Deactivate virtual environment if activated
deactivate

echo "Job completed!"
