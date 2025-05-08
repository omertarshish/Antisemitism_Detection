#!/bin/bash
#SBATCH --job-name=phi4-inference
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --output=%j.out

# Get server address from ollama job output
OLLAMA_SERVER=$(grep "Ollama server running" ${OLLAMA_JOB_OUT} | awk '{print $NF}')

# Run inference
python ${PROJECT_ROOT}/scripts/run_inference.py \
  --model "phi:4" \
  --ip-port ${OLLAMA_SERVER} \
  --input ${PROJECT_ROOT}/data/raw/tweets.csv \
  --output ${PROJECT_ROOT}/data/results/phi4_results.csv \
  --batch-size 50 \
  --definitions IHRA,JDA