#! /bin/bash

#SBATCH --mem-per-cpu=16G
#SBATCH --time=20:00:00
#SBATCH -J marc.fassbender@rwth-aachen.de
#SBATCH -o %j.out

#SBATCH rwth0776

source ~/anaconda3/bin/activate
conda activate snpnet

python ~/snpnet/cluster_scripts/BCE.py $1 $2