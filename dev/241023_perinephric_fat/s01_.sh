#!/bin/sh

#SBATCH -J manual_review_queue
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --mem 64000

#SBATCH --output=dev/241023_perinephric_fat/logs/01_out.txt
#SBATCH --error=dev/241023_perinephric_fat/logs/01_err.txt

#SBATCH -p defq


module load python/gpu/3.10.6
source /home/hellern/isilon/env/a100-3.10-v2/bin/activate
module list

python3 /home/hellern/isilon/code/repos/seg-to-nephrometry/241023_perinephric_fat/s01_prelim.py
