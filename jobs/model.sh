#!/bin/bash
# FILEPATH: /home/scc/mw8007/blocking/jobs/model.sh

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-model
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --output="/home/scc/mw8007/blocking/jobs/model-era5-era5-msl.out"
#SBATCH --error="/home/scc/mw8007/blocking/jobs/model-era5-era5-msl.error"

cd /home/scc/mw8007/blocking

poetry shell

mpiexec -n 1 ./.venv/bin/python model/model.py