#!/bin/bash
# FILEPATH: /home/scc/mw8007/blocking/jobs/model.sh

#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=blocking
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1

module restore blocking
cd /home/scc/mw8007/blocking
poetry shell

python model/model.py