#!/bin/bash
#SBATCH --partition=batch-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/distributed.py 2