import subprocess
import torch
import time
import re
import math
from typing import List, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
import torch.multiprocessing as mp
import os

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def collective_operations_main(rank: int, world_size: int, content_path: str):
    """Try out some collective operations."""
    # Note: this function is running asynchronously for each process (world_size)

    setup(rank, world_size, content_path)

    # All-reduce
    """if rank == 0:
        note("### All-reduce")"""
    dist.barrier()  # Waits for all processes to get to this point

    tensor = torch.tensor([0., 1, 2, 3], device=f"cuda:{rank}") + rank  # Both input and output

    #note(f"Rank {rank} [before all-reduce]: {tensor}", verbatim=True)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    #note(f"Rank {rank} [after all-reduce]: {tensor}", verbatim=True)
    
    # Reduce-scatter
    """if rank == 0:
        note("### Reduce-scatter")"""
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=f"cuda:{rank}") + rank  # Input
    output = torch.empty(1, device=f"cuda:{rank}")  # Allocate output

    #note(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", verbatim=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    #note(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", verbatim=True)

    # All-gather
    """if rank == 0:
        note("### All-gather")"""
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=f"cuda:{rank}")  # Allocate output

    #note(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", verbatim=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    #note(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", verbatim=True)

    """if rank == 0:
        note("Recall that all-reduce = reduce-scatter + all-gather!")
    
    cleanup()"""

if __name__ == "__main__":
    mp.spawn(fn=collective_operations_main, args=(2, "gloo"))