import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import timeit

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, vector_size, backend, n_warmup, result):
    setup(rank, world_size, backend)
    if torch.cuda.is_available():
        data = torch.randint(0, 10, (vector_size,)).to("cuda")
    else:
        data = torch.randint(0, 10, (vector_size,))
    #print(f"rank {rank} data (before all-reduce): {data}")
    for _ in range(n_warmup):
        dist.all_reduce(data, async_op=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    end_time = timeit.default_timer()
    timing = (end_time - start_time) * 1e3
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    result[rank] = timing
    return timing

def benchmark(func, n_runs, n_warmup=5):
    timings = []
    for _ in range(n_runs):
        time = func()
        timings.append(time)
    return timings


if __name__ == "__main__":
    #world_size = int(sys.argv[1])
    parameters = {
        "vector_size" : [1000, 1e6, 10e6, 50e6, 100e6, 500e6, 1e9],
        #"vector_size" : [1e9],
        "world_size" : [2, 4, 6],
        "backend" : ["gloo"]
    }
    for backend in parameters["backend"]:
        print(f"Backend: {backend}")
        for x in parameters["world_size"]:
            print(f"World size: {x}")
            for y in parameters["vector_size"]:
                print(f"Vector size: {y}")
                vector_size_divided = int(y // 4)
                timing_dict = []
                result = torch.zeros(x)
                timing_dict = benchmark(lambda: mp.spawn(fn=distributed_demo, args=(x, vector_size_divided, backend, 5, result), nprocs=x, join=True), 1)
                print(f"Timings: {result}")
                print(f"Average time: {sum(result) / len(result)} milliseconds")
        """print(f"World size: {world_size}")
        vector_size = parameters["vector_size"][int(sys.argv[2])]
        #for vector_size in parameters["vector_size"]:
        print(f"Vector size: {vector_size}")
        vector_size_divided = int(vector_size // 4)
        timing_dict = []
        #divide by four because float32 is 4 bytes
        result = torch.zeros(world_size)
        timing_dict = benchmark(lambda: mp.spawn(fn=distributed_demo, args=(world_size, vector_size_divided, backend, 5, result), nprocs=world_size, join=True), 1)
        #timing_dict = torch.tensor(timing_dict)  # Convert to tensor for all_gather_object
        #all_timings = [torch.zeros(world_size) for _ in range(5)]  # Create a list to store timings from all ranks
        #dist.all_gather_object(all_timings, timing_dict)  # Gather timings from all ranks
        #avg_timings = [torch.mean(t).item() for t in all_timings]  # Compute average timings
        #print(f"Timings: {avg_timings}")
        #print(f"Average time: {sum(avg_timings) / len(avg_timings)} milliseconds")
        print(f"Timings: {result}")
        print(f"Average time: {sum(result) / len(result)} milliseconds")"""
