#!/bin/bash
# FILEPATH: /home/scc/mw8007/blocking/jobs/propulate.sh

#SBATCH --ntasks=8
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-propulate
#SBATCH --partition=normal
#DEBUG --gres=gpu:4g.20gb:4
#SBATCH --gres=gpu:full:4
#SBATCH --output="/home/scc/mw8007/blocking/jobs/propulate-era5-hyperparameter-msl.out"
#SBATCH --error="/home/scc/mw8007/blocking/jobs/propulate-era5-hyperparameter-msl.error"

cd /home/scc/mw8007/blocking

poetry shell

mpiexec -n 8 ./.venv/bin/python model/prop.py