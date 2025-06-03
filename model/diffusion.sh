#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J diffusion
#SBATCH --gpus=1  

module load miniconda/24.9.2
source activate zyf

python -u diffusion.py