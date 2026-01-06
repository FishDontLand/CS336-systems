import argparse

import cs336_basics.module as module
import cs336_basics.optimizer as optimizer
import torch
import numpy as np
import timeit
import torch.cuda.nvtx as nvtx

from cs336_basics.utils import cross_entropy

def monitor_transformer(vocab_size: int, d_model: int, d_ff: int, num_layers: int, num_heads: int,
                        theta: float, context_length: int, batch_size: int, warmup_steps:int,
                        num_measures: int, forward_only: bool=False):
    print("Model Params: ", {"d_model": d_model, "d_ff": d_ff, "num_layers": num_layers, "num_heads": num_heads})
    device = torch.device("cuda:0")
    lm = module.TransformerLM(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, theta, device=device)

    rng = np.random.default_rng()
    random_data = rng.integers(0, vocab_size, size=(batch_size, context_length + 1))

    batch_input = torch.tensor(random_data[:, :-1], dtype=torch.long, device=device)
    batch_target = torch.tensor(random_data[:, 1:], dtype=torch.long, device=device)

    optim = optimizer.AdamW(lm.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1)
    for _ in range(warmup_steps):
        optim.zero_grad()
        output = lm(batch_input)
        if not forward_only:
            loss = cross_entropy(output, batch_target)
            loss.backward()

    forward_times = []
    backward_times = []
    torch.cuda.synchronize()

    for _ in range(num_measures):
        optim.zero_grad()
        start = timeit.default_timer()
        with nvtx.range("forward pass"):
            output = lm(batch_input)
            torch.cuda.synchronize()
        end = timeit.default_timer()

        forward_times.append(end - start)
        if not forward_only:
            loss = cross_entropy(output, batch_target)
            start = timeit.default_timer()
            with nvtx.range("backward pass"):
                loss.backward()
                torch.cuda.synchronize()
            end = timeit.default_timer()
            backward_times.append(end - start)
            with nvtx.range("param update"):
                optim.step()

    print("forward pass times")
    print(forward_times)
    print("mean: ", np.mean(forward_times), " std: ", np.std(forward_times))

    if not forward_only:
        print("backward pass times")
        print(backward_times)
        print("mean: ", np.mean(backward_times), " std: ", np.std(backward_times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--forward_only", type=bool, default=False)
    parser.add_argument("--num_measures", type=int, required=True)

    args = parser.parse_args()
    monitor_transformer(**vars(args))

