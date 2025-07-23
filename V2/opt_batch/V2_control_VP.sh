#!/bin/bash
#SBATCH --account=p32750  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=short  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=16 ## how many cpus or processors do you need on each computer
#SBATCH --time=02:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=8G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=V2_control_VP  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --output=V2_control_VP ## standard out and standard error goes to this file


module purge all
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate /home/yuj1135/.conda/envs/elephant

python /projects/p32750/repository/Natural_motion-model/V2_control_VP.py 