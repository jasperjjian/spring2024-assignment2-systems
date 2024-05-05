#!/bin/bash
#SBATCH --partition=a2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --time=00:01:00

SIZE=$1
echo "SIZE: ${SIZE}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate cs336_systems

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
# Execute command for each task
srun python3 cs336-systems/cs336_systems/multinode_distributed.py $SIZE