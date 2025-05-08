#!/bin/bash
################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
################################################################################################
#SBATCH --partition gpu                     ### Partition name for GPU resources
#SBATCH --job-name phi4-inference           ### Name of the job
#SBATCH --output output_logs/job-%J.out     ### Output log for running job - %J is the job number
#SBATCH --gpus=rtx_6000:1                  ### Request 1 RTX 6000 GPU
#SBATCH --mem=24G                          ### Memory allocation
#SBATCH --time=24:00:00                    ### Time limit (7 days max: D-H:MM:SS)
##SBATCH --mail-user=omertar@post.bgu.ac.il   ### User's email for notifications (uncomment to use)
##SBATCH --mail-type=BEGIN,END,FAIL        ### Notification conditions (uncomment to use)

# Model information
MODEL_NAME="phi4:14b"                          # The Phi-4 model name in Ollama

### Print job information
echo "Starting inference job with model: ${MODEL_NAME}"
echo "Date: $(date)"
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST
echo -e "Ollama endpoint:\t\t" $OLLAMA_IP_PORT

### Validate required environment variable
if [ -z "$OLLAMA_IP_PORT" ]; then
    echo "ERROR: OLLAMA_IP_PORT environment variable not set!"
    echo "Please use: sbatch --export=OLLAMA_IP_PORT=<IP:PORT> environments/slurm/submit_scripts/phi4_inference.sbatch"
    exit 1
fi

### Create result directories
mkdir -p data/results/phi4
mkdir -p data/results/phi4/batches

# Set custom temp directory path in home directory (to avoid permission issues)
export ANTISEMITISM_TEMP_DIR="${HOME}/temp_data/batches"
mkdir -p ${ANTISEMITISM_TEMP_DIR}

### Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=TRUE

### Load the anaconda module and activate environment
module load anaconda
source activate tweet-analysis  # Replace with your actual environment name if different

### Run inference with the model
echo "Running inference with ${MODEL_NAME} on server ${OLLAMA_IP_PORT}"

python scripts/run_inference.py \
  --model "${MODEL_NAME}" \
  --ip-port "${OLLAMA_IP_PORT}" \
  --input data/raw/tweets.csv \
  --output data/results/phi4/analysis_results.csv \
  --batch-size 50 \
  --definitions IHRA,JDA \
  --config ollama_config

echo "Job completed at $(date)"