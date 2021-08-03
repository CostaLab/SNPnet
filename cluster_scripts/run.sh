#! /bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --account=rwth0776
#SBATCH -J marc.fassbender@rwth-aachen.de
#SBATCH -o ./outputs/%j.out

#SBATCH --gres=gpu:1
 
module load cuda

source ~/anaconda3/bin/activate
conda activate snpnet

python ~/snpnet/cluster_scripts/BCE.py $1