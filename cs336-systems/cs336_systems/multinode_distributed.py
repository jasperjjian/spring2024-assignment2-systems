import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from datetime import timedelta
import sys
import timeit

def setup(backend):
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank, local_world_size

def multinode_distributed_demo(backend, vector_size, result):
    rank, world_size, local_rank, local_world_size = setup(backend)
    print(
    f"World size: {world_size}, global rank: {rank}, "
    f"local rank: {local_rank}, local world size: {local_world_size}"
    )
    if torch.cuda.is_available():
        data = torch.ones(vector_size).to("cuda")
    else:
        data = torch.ones(vector_size)
    #print(f"rank {rank} data (before all-reduce): {data}")
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    end_time = timeit.default_timer()
    timing = (end_time - start_time) * 1e3
    result[rank + local_rank] = timing
    #print(f"rank {rank} data (after all-reduce): {data}")
    return

def reset_distributed_process():
    # Check if the process is part of a distributed group
    if dist.is_initialized():
        # Cleanup the existing distributed group
        dist.destroy_process_group()


if __name__ == "__main__":
    parameters = {
        "vector_size" : [1000, 1e6, 10e6, 50e6, 100e6, 500e6, 1e9],
        "backend" : ["nccl"]
    }
    for backend in parameters["backend"]:
        print(f"Backend: {backend}")
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        world_size = int(os.environ["SLURM_NTASKS"])
        print(int(sys.argv[1]))
        y = parameters["vector_size"][int(sys.argv[1])]
        print(f"Vector size: {y}")
        vector_size_divided = int(y // 4)
        result = torch.zeros(world_size * local_world_size)        
        multinode_distributed_demo(backend, vector_size_divided, result)
        print(f"Timings: {result}")
        print(f"Average time: {sum(result) / len(result)} milliseconds")