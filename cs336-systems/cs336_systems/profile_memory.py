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

def benchmark_memory(model, input, target, loss, warm_up, trials, backward=False):
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
            preds = model(input)
            if backward:
                l = loss(preds, target)
                l.backward()
                torch.cuda.synchronize
            prof.step()
            prof.export_memory_timeline("timeline.html", device=device)

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    return 