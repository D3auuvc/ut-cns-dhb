#!/bin/bash
#SBATCH -J CNS
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 00:45:00
#re = '^[0-9]+\.[0]+$'

module load cuda/10.0
cd $HOME/CNS/
./training.py
