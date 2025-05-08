#!/bin/bash
#SBATCH --job-name=ollama-server
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=%j.out

module load apptainer

# Start Ollama server
bash ${PROJECT_ROOT}/environments/slurm/scripts/start_ollama.sh