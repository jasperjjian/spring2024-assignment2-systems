#!/bin/bash
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=24G

#accept one argument which indexes the correct size vector

SIZE=$1

python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/distributed.py 4 $SIZE