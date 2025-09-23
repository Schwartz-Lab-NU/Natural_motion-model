#!/bin/bash
#SBATCH --account=p32750  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=normal  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=24 ## how many cpus or processors do you need on each computer
#SBATCH --time=16:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=3G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=V3_natural_counts_gaussianstep1  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --output=V3_natural_counts_gaussianstep1 ## standard out and standard error goes to this file

module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate /home/yuj1135/.conda/envs/elephant

# Set working directory
cd /projects/p32750/repository/Natural_motion-model/V3/opt_batch



# Print job information
echo "Starting V3 Natural counts Gaussian Step1 Optimization"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

# Run the optimization
python V3_natural_counts_gaussianstep1.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "V3 Natural counts optimization completed successfully"
    echo "Results saved to: /projects/p32750/repository/Natural_motion-model/V3/results/V3_natural_counts_gaussianstep1.pkl"
else
    echo "V3 Natural counts optimization failed with exit code $?"
    exit 1
fi

echo "Job completed at: $(date)" 