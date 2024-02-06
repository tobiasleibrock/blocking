#!/bin/bash
# FILEPATH: /home/scc/mw8007/blocking/jobs/propulate.sh

#SBATCH --ntasks=64
#SBATCH --time=8:00:00
#SBATCH --job-name=blocking-propulate
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:4
#SBATCH --output="/home/scc/mw8007/blocking/jobs/propulate.out"
#SBATCH --error="/home/scc/mw8007/blocking/jobs/propulate.error"

cd /home/scc/mw8007/blocking

poetry shell

mpiexec -n 64 ./.venv/bin/python model/prop.py