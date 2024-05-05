#!/bin/bash

# iterates over integers [0-7] and calls a bash script

for i in {0..6}; do
    sbatch scripts/run_distributed_multinode_1.sh $i
done