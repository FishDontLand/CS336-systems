import copy
import timeit
import os
import einx
import numpy as np
import torch
import triton
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.module import MultiheadSelfAttention, TransformerLM
from cs336_basics.optimizer import AdamW

from cs336_basics.utils import scaled_dot_product_attention

from cs336_systems.FlashAttention import TritonFlashAttention
from cs336_systems.monitoring import monitor_transformer
import pandas as pd

from cs336_systems.ddp import DDP
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
    num_runs = 100
    # code for debug purpose
    # num_runs = 20
    batch_size = 16
    torch.manual_seed(42)
    d_model_lst = [16, 32, 64, 128]
    context_length_lst = [256, 1024, 4096, 8192, 16384]
    # below is the code for debug purpose
    # d_model_lst = [4, 8]
    # context_length_lst = [16, 32]
    forward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
    backward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]


    for d_model_idx, d_model in enumerate(d_model_lst):
        for context_len_idx, context_length in enumerate(context_length_lst):
            print(f"Working on d_model: {d_model}, context_length: {context_length}")
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
                    if compile_layer:
                        save_path = f"../outputs/memory_logs/compiled/d_model_{d_model}_ctx_len_{context_length}_forward_memory.pickle"
                    else:
                        save_path = f"../outputs/memory_logs/d_model_{d_model}_ctx_len_{context_length}_forward_memory.pickle"
                    torch.cuda.memory._dump_snapshot(save_path)
                    torch.cuda.memory._record_memory_history(enabled=None)

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
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

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
                print(e)
                if torch.cuda.is_available():
                    torch.cuda.memory._record_memory_history(enabled=None)
            finally:
                print("="* 20, "Finished", '=' * 20)

    column_idx = pd.Index(context_length_lst, name="context_length")
    row_idx = pd.Index(d_model_lst, name="d_model")

    forward_time_table = pd.DataFrame(forward_time_table, index=row_idx, columns=column_idx)
    backward_time_table = pd.DataFrame(backward_time_table, index=row_idx, columns=column_idx)

    print("forward time")
    print(forward_time_table)

    print("backward time")
    print(backward_time_table)

def vanilla_attn(q, k, v):
    seq_len = q.size(-2)
    masks = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
    masks = torch.triu(masks).transpose(0, 1)
    masks = einx.logical_and('..., q_len k_len -> ... q_len k_len',
                             torch.ones(k.size()[:-2], dtype=torch.bool, device=q.device), masks)

    output = scaled_dot_product_attention(q, k, v, masks)
    return output

def benchmarking_flash():
    device = torch.device('cuda:0')
    warmup_steps = 10
    num_runs = 100
    batch_size = 1
    torch.manual_seed(34)
    d_model_lst = [16, 32, 64, 128]
    context_length_lst = [2 ** i for i in range(7, 17)]
    dtype_lst = [torch.float32, torch.bfloat16]

    for dtype in dtype_lst:
        vanilla_forward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
        vanilla_full_pass_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
        flash_forward_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
        flash_full_pass_time_table = [[None] * len(context_length_lst) for _ in range(len(d_model_lst))]
        for d_model_idx, d_model in enumerate(d_model_lst):
            for context_len_idx, context_len in enumerate(context_length_lst):
                Q = torch.randn(batch_size, context_len, d_model, device=device, dtype=dtype, requires_grad=True)
                K = torch.randn(batch_size, context_len, d_model, device=device, dtype=dtype, requires_grad=True)
                V = torch.randn(batch_size, context_len, d_model, device=device, dtype=dtype, requires_grad=True)
                dO = torch.randn(batch_size, context_len, d_model, device=device, dtype=dtype)

                try:
                    time = triton.testing.do_bench(lambda: vanilla_attn(Q, K, V), warmup=warmup_steps,
                                                 rep=num_runs)
                except:
                    time = float('inf')

                vanilla_forward_time_table[d_model_idx][context_len_idx] = time

                try:
                    time = triton.testing.do_bench(lambda: TritonFlashAttention.apply(Q, K, V, True),
                                                   warmup=warmup_steps, rep=num_runs)
                except:
                    time=float('inf')

                flash_forward_time_table[d_model_idx][context_len_idx] = time

                try:
                    time = triton.testing.do_bench(lambda: vanilla_attn(Q, K, V).backward(dO), warmup=warmup_steps,
                                                   rep=num_runs, grad_to_none=[Q, K, V])
                except:
                    time = float('inf')

                vanilla_full_pass_time_table[d_model_idx][context_len_idx] = time

                try:
                    time = triton.testing.do_bench(lambda: TritonFlashAttention.apply(Q, K, V, True),
                                                   warmup=warmup_steps, rep=num_runs, grad_to_none=[Q, K, V])
                except:
                    time = float('inf')

                flash_full_pass_time_table[d_model_idx][context_len_idx] = time

        vanilla_forward_time_table = pd.DataFrame(vanilla_forward_time_table,
                                                  index=pd.Index(d_model_lst, name="d_model"),
                                                  columns=pd.Index(context_length_lst, name="context_length"))

        flash_forward_time_table = pd.DataFrame(flash_forward_time_table,
                                                index=pd.Index(d_model_lst, name="d_model"),
                                                columns=pd.Index(context_length_lst, name="context_length"))

        vanilla_full_pass_time_table = pd.DataFrame(vanilla_full_pass_time_table,
                                                    index=pd.Index(d_model_lst, name="d_model"),
                                                    columns=pd.Index(context_length_lst, name="context_length"))

        flash_full_pass_time_table = pd.DataFrame(flash_full_pass_time_table,
                                                  index=pd.Index(d_model_lst, name="d_model"),
                                                  columns=pd.Index(context_length_lst, name="context_length"))

        forward_time_table = pd.concat({'vanilla': vanilla_forward_time_table,
                                        'flash_attn': flash_forward_time_table}, axis=1)
        forward_time_table = forward_time_table.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        full_pass_time_table = pd.concat({'vanilla': vanilla_full_pass_time_table,
                                          'flash_attn': flash_full_pass_time_table}, axis=1)
        full_pass_time_table = full_pass_time_table.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        backward_time_table = full_pass_time_table - forward_time_table

        forward_time_table.to_csv(f'../outputs/benchmark/forward_pass_{dtype}.csv')
        backward_time_table.to_csv(f'../outputs/benchmark/backward_pass_{dtype}.csv')
        full_pass_time_table.to_csv(f'../outputs/benchmark/full_pass_{dtype}.csv')


def setup(rank: int, world_size: int, device_name: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    if device_name == "cuda" and torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def all_reduce(rank: int, world_size: int, num_elements: int, device_name: str):
    setup(rank, world_size, device_name)
    tensor = torch.randn(num_elements, dtype=torch.float32)
    warm_up_steps = 5
    if device_name == 'cuda' and torch.cuda.is_available():
        tensor = tensor.to(device_name)
    for _ in range(warm_up_steps):
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)

    if device_name == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    start_time = timeit.default_timer()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if device_name == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    end_time = timeit.default_timer()

    duration = end_time - start_time
    print(f'rank {rank} duration: {duration:4f} seconds')
    all_durations = [None for _ in range(world_size)]
    dist.all_gather_object(all_durations, duration)
    avg = sum(all_durations) / len(all_durations)
    print(f"rank {rank}: mean duration {avg:4f} seconds")


def benchmark_all_reduce(mode="cpu"):
    for world_size in [2, 4, 6]:
        print("working with world size", world_size)
        if mode=="gpu" and torch.cuda.is_available() and torch.cuda.device_count() < world_size:
            print('skipped world size', world_size)
            continue
        for data_byte_size in [10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9]:
            num_mb = round(data_byte_size / 1e6)
            print(f"data size: {num_mb:0f} MB")
            num_elements = data_byte_size // 8
            mp.spawn(fn=all_reduce, args=(world_size, num_elements, 'cpu'), nprocs=world_size, join=True)
        print("=" * 30)

def benchmark_dist_train_model(fn, additional_args: tuple, device_type: str='cpu'):
    vocab_size = 10000
    # d_model = 1600
    # num_layers = 48
    # num_heads = 25
    # d_ff = 6400
    # context_length = 512

    # toy model params for cpu testing
    d_model = 16
    num_layers = 4
    num_heads = 4
    d_ff = 32
    context_length = 512

    theta=10000
    world_size = 2
    batch_size = 16

    rng = np.random.default_rng()
    random_data = torch.tensor(rng.integers(0, vocab_size, size=(batch_size, context_length)),
                               dtype=torch.long)

    if device_type == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device = torch.device(device_type)
    elif device_type != 'cpu':
        print("current device type not supported, switched to cpu mode")
        device = torch.device(device_type)
        device_type = 'cpu'
    else:
        device = torch.device(device_type)

    lm = TransformerLM(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, theta, device=device)
    args = (world_size, 5, lm, random_data, device_type) + additional_args
    mp.spawn(fn=fn, args=args, nprocs=world_size, join=True)


def dist_training_with_ddp(rank: int, world_size: int, warmup_steps: int, model: torch.nn.Module, data: torch.tensor,
                           device_type: str):
    setup(rank, world_size, device_type)
    batch_size = data.size(0) // world_size
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    data_batch = data[start_idx:end_idx]
    data_batch = data_batch.to(device_type)
    assert len(data_batch) == batch_size

    rank_model = copy.deepcopy(model)
    rank_model = DDP(rank_model.to(device_type))
    optimizer = AdamW(rank_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    all_run_time = [None for _ in range(world_size)]
    for i in range(warmup_steps + 1):
        if i == warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            start_time = timeit.default_timer()

        optimizer.zero_grad()
        output = rank_model(data_batch)
        loss = (output * output).mean()
        loss.backward()
        rank_model.finish_gradient_synchronization()
        optimizer.step()

        if i == warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            end_time = timeit.default_timer()
            dist.all_gather_object(all_run_time, end_time - start_time)

    if rank == 0:
        avg_time = sum(all_run_time) / len(all_run_time)
        print(f'average run time: {avg_time}')

if __name__ == "__main__":
    benchmark_dist_train_model(dist_training_with_ddp, ())
    print('finished')

