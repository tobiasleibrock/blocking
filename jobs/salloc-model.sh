#!/bin/bash
# FILEPATH: /home/scc/mw8007/blocking/jobs/model.sh

salloc --ntasks=1 --time=3:00:00 --job-name=blocking --partition=normal --gres=gpu:full:1

module restore blocking
cd /home/scc/mw8007/blocking
poetry shell

python model/model.py