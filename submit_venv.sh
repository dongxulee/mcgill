#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=20G      # increase as needed
#SBATCH --time=1:00:00

source $HOME/jupyter_py3/bin/activate
python test2.py
