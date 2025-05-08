#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main                        ### specify partition name where to run a job.
#SBATCH --job-name output_logs/ollama-job                   ### name of the job
#SBATCH --output job-%J.out                     ### output log for running job - %J for job number
#SBATCH --gpus=rtx_6000:1

# Note: the following 4 lines are commented out
##SBATCH --mail-user=omertar@bgu.ac.il            ### user's email for sending job status messages
##SBATCH --mail-type=ALL                 ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=48G                               ### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
echo -e "IP ADDRESS:" $(hostname -i) "\n\n"

### Start your code below ####
./environments/slurm/submit_scripts/apptainer-ollama.sh