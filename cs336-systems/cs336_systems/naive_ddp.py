import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import timeit
from datetime import timedelta
import torch.nn as nn
import torch
import cs336_systems.toymodel as toymodel
from cs336_basics.nn_utils import cross_entropy
import cs336_basics.model
import cs336_basics.nn_utils
import cs336_systems.benchmark

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

def ddp_main(data: torch.Tensor, model : nn.Module, model_args : tuple, loss_fn, num_steps: int, backend: str):
    torch.random.manual_seed(0)
    train_timings = []
    communication_timings = []
    rank, world_size, local_rank, local_world_size = setup(backend)

    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(f"cuda")

    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    model_instance = model(*model_args).to(f"cuda")
    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=1e-3)
    loss_fn = loss_fn
    # Calculate time for training total and all reduce separately
    for _ in range(num_steps):
        start_time_train = timeit.default_timer()
        # Forward pass
        x = data
        output = model_instance(x)
        loss = loss_fn(output, x)
        # Backward pass
        loss.backward()
        start_time_communication = timeit.default_timer()
        # Sync gradients across workers (NEW!)
        for param in model_instance.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        end_time_communication = timeit.default_timer()
        # Update parameters
        optimizer.step()
        end_time_train = timeit.default_timer()
        train_timings.append((end_time_train - start_time_train) * 1e3)
        communication_timings.append((end_time_communication - start_time_communication) * 1e3)
    print(f"Rank {rank}: loss = {loss.item()}, params = {[summarize_tensor(param) for param in model_instance.parameters()]}")
    return model_instance, train_timings, communication_timings

def ddp_main_batched_transfer(data: torch.Tensor, model : nn.Module, model_args : tuple, loss_fn, num_steps: int, backend: str):
    torch.random.manual_seed(0)
    train_timings = []
    communication_timings = []
    
    rank, world_size, local_rank, local_world_size = setup(backend)

    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(f"cuda")

    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    model_instance = model(*model_args).to(f"cuda")
    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=1e-3)
    #loss_fn = loss_fn()
    # Calculate time for training total and all reduce separately
    for _ in range(num_steps):
        start_time_train = timeit.default_timer()
        # Forward pass
        x = data
        output = model_instance(x)
        loss = loss_fn(output, x)
        # Backward pass
        loss.backward()
        start_time_communication = timeit.default_timer()
        # Sync gradients across workers (NEW!)
        flattened_grads = torch.cat([param.grad.view(-1) for param in model_instance.parameters()])
        dist.all_reduce(tensor=flattened_grads, op=dist.ReduceOp.AVG)
        end_time_communication = timeit.default_timer()
        # Reshape and update gradients
        start_idx = 0
        for param in model_instance.parameters():
            param.grad.data = flattened_grads[start_idx:start_idx+param.numel()].view_as(param.grad.data)
            start_idx += param.numel()
        # Update parameters
        optimizer.step()
        end_time_train = timeit.default_timer()
        train_timings.append((end_time_train - start_time_train) * 1e3)
        communication_timings.append((end_time_communication - start_time_communication) * 1e3)
    print(f"Rank {rank}: loss = {loss.item()}, params = {[summarize_tensor(param) for param in model_instance.parameters()]}")
    return model_instance, train_timings, communication_timings

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def main():
    # NAIVE TRAINING AND BENCHMARKING
    torch.random.manual_seed(0)
    backend = "nccl"
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    world_size = int(os.environ["SLURM_NTASKS"])
    
    final_model, train_time, communication_time = ddp_main_batched_transfer(torch.randn(100, 100), toymodel.ToyModel, (100, 100), nn.MSELoss, 100, backend)

    #print([parameter for parameter in final_model.parameters()])
    print("Train time: ", train_time)
    print("Average train time: ", sum(train_time)/len(train_time))
    print("Communication time: ", communication_time)
    print("Average communication time: ", sum(communication_time)/len(communication_time))

if __name__ == "__main__":
    model_configs = {
    "small" : {
        "d_model" : 768,
        "d_ff" : 3072,
        "num_layers" : 12,
        "num_heads" : 12
    },
    "medium" : {
        "d_model" : 1024,
        "d_ff" : 4096,
        "num_layers" : 24,
        "num_heads" : 16
    },
    "large" : {
        "d_model" : 1280,
        "d_ff" : 5120,
        "num_layers" : 36,
        "num_heads" : 20
    },
    "xl" : {
        "d_model" : 1600,
        "d_ff" : 6400,
        "num_layers" : 48,
        "num_heads" : 25
    },
    "2.7b" : {
        "d_model" : 2560,
        "d_ff" : 10240,
        "num_layers" : 32,
        "num_heads" : 32
    }}
    size_configs = model_configs["2.7b"]
    print("Model size: 2.7b")
    vocab_size = 10000
    context_length = 128
    batch_size = 16
    model = cs336_basics.model.BasicsTransformerLM
    
    input, target = cs336_systems.benchmark.generate_random_sample(context_length, vocab_size, batch_size, device="cuda:0")
    loss = cs336_basics.nn_utils.cross_entropy

    torch.random.manual_seed(0)
    backend = "nccl"
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    world_size = int(os.environ["SLURM_NTASKS"])

    final_model, train_time, communication_time = ddp_main_batched_transfer(input, model, (vocab_size, context_length, size_configs["d_model"], size_configs["num_layers"], size_configs["num_heads"], size_configs["d_ff"]), loss, 100, backend)

    #print([parameter for parameter in final_model.parameters()])
    print("Train time: ", train_time)
    print("Average train time: ", sum(train_time)/len(train_time))
    print("Communication time: ", communication_time)
    print("Average communication time: ", sum(communication_time)/len(communication_time))
