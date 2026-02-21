import timeit

import numpy as np
import torch

from cs336_basics.module import MultiheadSelfAttention
from cs336_systems.monitoring import monitor_transformer
import pandas as pd

from experiments.params import PARAM_MAP


def print_title(title, boundary_char_len):
    print("=" * boundary_char_len)
    print("=" * boundary_char_len)
    lef_boundary_len = (boundary_char_len - len(title)) // 2
    right_boundary_len = boundary_char_len - len(title) - lef_boundary_len
    print("=" * lef_boundary_len, title, right_boundary_len)
    print("=" * boundary_char_len)
    print("=" * boundary_char_len)

def benchmark_model(compile_model:bool=False):
    FIXED_ARGS = {
        "vocab_size": 10000,
        "theta": 10000,
        "batch_size": 4,
        "num_measures": 10,
    }
    benchmark_result = []
    warmup_step_lst = [5, 0, 1, 2]
    for warmup in warmup_step_lst:
        row = []
        for model_size, size_param in PARAM_MAP.items():
            print(f"Benchmarking {model_size} with warmup={warmup} ...")
            try:
                mean_time = monitor_transformer(**size_param, **FIXED_ARGS, warmup_steps=warmup,
                                                context_length=256, compile_model=compile_model)['forward']['mean']
            except Exception as e:
                print(e)
                mean_time = None
            row.append(mean_time)
            print("finished")
            print()

        benchmark_result.append(row)
    benchmark_result = pd.DataFrame(benchmark_result, columns=list(PARAM_MAP.keys()))
    benchmark_result['warmup_steps'] = warmup_step_lst
    benchmark_result = benchmark_result.set_index('warmup_steps')
    print(benchmark_result)

def benchmark_attention(compile_layer: bool=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    warmup_steps = 10
    # num_runs = 100
    num_runs = 20
    batch_size = 8
    torch.manual_seed(42)
    # d_model_lst = [16, 32, 64, 128]
    # context_length_lst = [256, 1024, 4096, 8192, 16384]
    d_model_lst = [4, 8]
    context_length_lst = [16, 32]
    forward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
    backward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]


    for d_model_idx, d_model in enumerate(d_model_lst):
        for context_len_idx, context_length in enumerate(context_length_lst):
            forward_run_times = []
            backward_run_times = []

            embedded_seq = torch.randn(batch_size, context_length, d_model).to(device)
            attn_layer = MultiheadSelfAttention(d_model, None, context_length, 10000, device)
            if compile_layer:
                attn_layer = torch.compile(attn_layer)
            try:
                if torch.cuda.is_available():
                    torch.cuda.memory._record_memory_history(max_entries=1000000)
                for _ in range(warmup_steps):
                    output = attn_layer(embedded_seq)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                for _ in range(num_runs):
                    start = timeit.default_timer()
                    output = attn_layer(embedded_seq)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end = timeit.default_timer()
                    forward_run_times.append(end - start)


                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.memory._dump_snapshot(f"d_model_{d_model}_ctx_len_{context_length}_forward_memory.pickle")
                    torch.cuda._record_memory_history(enabled=None)

                for _ in range(warmup_steps):
                    output = attn_layer(embedded_seq)
                    result = output.sum()
                    result.backward()
                    attn_layer.zero_grad()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                for _ in range(num_runs):
                    output =  attn_layer(embedded_seq)
                    result = output.sum()

                    start = timeit.default_timer()
                    result.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end = timeit.default_timer()
                    backward_run_times.append(end - start)

                    attn_layer.zero_grad()

                forward_time_table[d_model_idx][context_len_idx] = np.mean(forward_run_times)
                backward_time_table[d_model_idx][context_len_idx] = np.mean(backward_run_times)

            except Exception as e:
                raise e
                if torch.cuda.is_available():
                    torch.cuda.memory_record_memory_history(enabled=None)
                print(e)

    column_idx = pd.Index(context_length_lst, name="context_length")
    row_idx = pd.Index(d_model_lst, name="d_model")

    forward_time_table = pd.DataFrame(forward_time_table, index=row_idx, columns=column_idx)
    backward_time_table = pd.DataFrame(backward_time_table, index=row_idx, columns=column_idx)

    print("forward time")
    print(forward_time_table)

    print("backward time")
    print(backward_time_table)


if __name__ == "__main__":
    benchmark_attention(True)

