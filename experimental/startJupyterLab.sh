#!/bin/bash
# Author: M.Jacobs

# Slurm directives
#SBATCH --job-name=JupyterLab
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb

cd $SLURM_SUBMIT_DIR

module purge
module load JupyterLab/2.2.8-GCCcore-10.2.0
module load SciPy-bundle/2020.11-foss-2020b
module load scikit-learn/0.23.2-foss-2020b
#module load PyTorch/1.9.0-foss-2020b
module load torchvision/0.8.2-foss-2020b-PyTorch-1.7.1
module load matplotlib/3.3.3-foss-2020b

hostnamectl
jupyter-lab
