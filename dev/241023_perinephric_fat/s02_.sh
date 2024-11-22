#!/bin/sh

#SBATCH -J manual_review_queue
#SBATCH -c 12
#SBATCH -n 1
#SBATCH --mem 32000

#SBATCH --output=dev/241023_perinephric_fat/logs/02_out.txt
#SBATCH --error=dev/241023_perinephric_fat/logs/02_err.txt

#SBATCH -p defq


module load python/gpu/3.10.6
source /home/hellern/isilon/env/a100-3.10-v2/bin/activate
module list

python3 /home/hellern/isilon/code/repos/seg-to-nephrometry/dev/241023_perinephric_fat/s02_aggregate.py
