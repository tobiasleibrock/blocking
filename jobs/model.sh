#!/bin/bash
# FILEPATH: blocking/model.sh

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-model
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --output="blocking/jobs/model.out"
#SBATCH --error="blocking/jobs/model.error"

cd blocking

poetry shell

mpiexec -n 1 ./.venv/bin/python model/model.py