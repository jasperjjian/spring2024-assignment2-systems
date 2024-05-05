import cs336_basics
import torch
import timeit
import cs336_basics.data
import argparse
import numpy as np
from torch.cuda.amp import autocast
from contextlib import nullcontext

import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_systems import kernels, naive_ddp, ddp_variants

def generate_random_sample(context_len, vocab_size, batch_size, device="cpu"):
    random_vector = np.random.randint(0, vocab_size, (context_len * 100,))
    return cs336_basics.data.get_batch(random_vector, batch_size, context_len, device)

def benchmark(model, input, target, loss, warm_up, trials, backward=False):
    final_times = []
    for _ in range(warm_up):
        preds = model(input)
    torch.cuda.synchronize
    for trial in range(trials):
        trial_times = []
        start = timeit.default_timer()
        preds = model(input)
        forward_time = timeit.default_timer() - start
        trial_times.append(forward_time)
        torch.cuda.synchronize

        if backward:
            start = timeit.default_timer()
            l = loss(preds, target)
            l.backward()
            backward_time = timeit.default_timer() - start
            trial_times.append(backward_time)
            torch.cuda.synchronize
        
        final_times.append(trial_times)
    return final_times

def benchmark_memory(model, input, target, loss, warm_up, trials, backward=False):
    optimizer = cs336_basics.optimizer.AdamW(model.parameters())
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    n_steps = 3
    with profile(
    activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            run_step(model, input, target, optimizer, loss, backward=backward)
            prof.step()
        
        prof.export_memory_timeline("timeline.html", device="cuda")

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    return 


def run_step(model, input, target, optimizer, loss, backward=True):
    with record_function('forward_pass'):
        preds = model(input)

    if backward:
        with record_function('backward_pass'):
            ce_loss = loss(preds, target)
            ce_loss.backward()
        with record_function('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return


def profile_model(model, input, target, optimizer, loss, n_steps, warmup, backward=True):

    for _ in range(warmup):
        preds = model(input)
    torch.cuda.synchronize

    with profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,], 
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    ) as prof:
        
        for _ in range(n_steps):
            run_step(model, input, target, optimizer, loss, backward=backward)
            prof.step()
            torch.cuda.synchronize
    
    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    return

def benchmark_rms(dim, warmup, device="cpu"):
    input_tensor = torch.randn(50000, dim, device=device)
    rms_times = []
    layernorm_times = []
    rms_triton_times = []
    rms = cs336_basics.model.RMSNorm(dim).to(device)
    layernorm = torch.nn.LayerNorm(dim).to(device)
    dy = torch.randn(50000, dim, device=device)
    
    for _ in range(warmup):
        output = rms(input_tensor)
        torch.cuda.synchronize

    for _ in range(1000):
        start = timeit.default_timer()
        output.grad = None
        output = rms(input_tensor)
        #output.backward(dy)
        total = (timeit.default_timer() - start) * 1000
        rms_times.append(total)
        torch.cuda.synchronize

    for _ in range(warmup):
        output = layernorm(input_tensor)
        torch.cuda.synchronize

    for _ in range(1000):
        start = timeit.default_timer()
        output.grad = None
        output = layernorm(input_tensor)
        #output.backward(dy)
        total = (timeit.default_timer() - start) * 1000
        layernorm_times.append(total)
        torch.cuda.synchronize
    
    
    weight = torch.randn(dim, device=device)
    for _ in range(warmup):
        output = kernels.rms_norm_triton.apply(input_tensor, weight)
        torch.cuda.synchronize

    for _ in range(1000):
        start = timeit.default_timer()
        output.grad = None
        kernels.rms_norm_triton.apply(input_tensor, weight)
        total = (timeit.default_timer() - start) * 1000
        rms_triton_times.append(total)
        torch.cuda.synchronize
        

    print(f"RMS Norm Mean: {np.mean(rms_times)}")
    print(f"RMS Norm Var: {np.var(rms_times)}")
    print("")
    print(f"LayerNorm Mean: {np.mean(layernorm_times)}")
    print(f"LayerNorm Var: {np.var(layernorm_times)}")
    print("")
    print(f"RMS Triton Mean: {np.mean(rms_triton_times)}")
    print(f"RMS Triton Var: {np.var(rms_triton_times)}")


    return

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
        }
    }
    parser = argparse.ArgumentParser()

    # Define flags
    parser.add_argument("--model_size", type=str)
    parser.add_argument("--warmup_runs", type=int)
    parser.add_argument("--timed_runs", type=int)
    parser.add_argument("--backward", type=str)
    parser.add_argument("--profile", type=str, default=False)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--norm", type=str, default="rms")

    # Parse arguments
    args = parser.parse_args()
    vocab_size = 10000
    context_length = 128
    batch_size = 16
    size_configs = model_configs[args.model_size]
    device = torch.device("cuda:0")
    model = cs336_basics.model.BasicsTransformerLM(vocab_size, context_length, size_configs["d_model"], size_configs["num_layers"], size_configs["num_heads"], size_configs["d_ff"], norm_type=args.norm).to(device)
    #torch.compile(model)
    input, target = generate_random_sample(context_length, vocab_size, batch_size, device="cuda:0")
    loss = cs336_basics.nn_utils.cross_entropy
    #trials = benchmark(model, input, target, loss, args.warmup_runs, args.timed_runs, eval(args.backward))
    benchmark_memory(model, input, target, loss, args.warmup_runs, args.timed_runs, eval(args.backward))

    print(args.model_size)
    sum = [sum(t) for t in trials]
    for i, t in enumerate(trials):
        print(f"Trial {i + 1}")
        if len(t) > 1:
            print(f"Forward: {t[0] * 1000}ms")
            print(f"Backward: {t[1] * 1000}ms")
        else:
            print(f"Forward: {t[0] * 1000}ms")
    print(f"Mean: {np.mean(sum) * 1000}ms")
    print(f"Variance: {np.var(sum) * 1000}ms")
    
    """if args.profile:
        model = cs336_basics.model.BasicsTransformerLM(vocab_size, context_length, size_configs["d_model"], size_configs["num_layers"], size_configs["num_heads"], size_configs["d_ff"]).to(device)
        if args.precision == 16:
            model = model.half()
        optimizer = cs336_basics.optimizer.AdamW(model.parameters())
        input, target = generate_random_sample(context_length, vocab_size, batch_size, device="cuda:0")
        loss = cs336_basics.nn_utils.cross_entropy
        profile_model(model, input, target, optimizer, loss, args.timed_runs, args.warmup_runs, backward=eval(args.backward))"""
    
    """for d in [1024, 2048, 4096, 8192]:
        print(f"Dimensionality: {d}")
        benchmark_rms(d, 1, device=device)"""
    
    

