#!/bin/bash
# FILEPATH: blocking/jobs/model.sh

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-model
#SBATCH --partition=normal
#SBAT --gres=gpu:full:1
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --output="jobs/model-ukesm+era5-test-era5-rnd.out"
#SBATCH --error="jobs/model-ukesm+era5-test-era5-rnd.error"

poetry shell

mpiexec -n 1 /home/scc/mw8007/blocking/.venv/bin/python model/model.py