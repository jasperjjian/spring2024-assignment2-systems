#!/bin/bash
#SBATCH --partition=a2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=24G

#python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark.py --model_size small --warmup_runs 1 --timed_runs 5 --backward False --norm triton --profile True
#python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark.py --model_size medium --warmup_runs 1 --timed_runs 5 --backward False --norm rms --profile True
#python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark.py --model_size large --warmup_runs 1 --timed_runs 5 --backward False --norm rms --profile True
#python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark.py --model_size xl --warmup_runs 1 --timed_runs 5 --backward False --norm rms --profile True
python3 -u /home/c-jjian/assignments/spring2024-assignment2-systems/cs336-systems/cs336_systems/benchmark.py --model_size 2.7b --warmup_runs 1 --timed_runs 5 --backward False --norm rms --profile True
