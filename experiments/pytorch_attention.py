import timeit

from cs336_basics import module, optimizer, utils
import torch
import numpy as np
import einx
import os

class VanillaAttention(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: int|None=None, theta: float|None=None, device: torch.device|None=None, dtype=None):
        super().__init__()
        self.q_proj = module.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = module.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = module.Linear(d_model, d_model, device=device, dtype=dtype)
        self.device=device
        if max_seq_len is None or theta is None:
            self.RoPE = None
        else:
            self.RoPE = module.RoPE(theta=theta, d_k=d_model, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(-2)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        token_positions = torch.arange(seq_len, device=x.device)
        Q = self.RoPE.forward(Q, token_positions)
        K = self.RoPE.forward(K, token_positions)


        masks = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
        masks = torch.tril(masks)
        masks = einx.logical_and('..., q_len k_len -> ... q_len k_len',
                         torch.ones(K.size()[:-2], dtype=torch.bool, device=self.device), masks)

        output = utils.scaled_dot_product_attention(Q, K, V, masks)
        return output



def profile_experiments(compile_attn=False):
    warmup_steps = 10
    device = torch.device("cuda:0")
    batch_size = 8

    for d_model in [16, 32, 64, 128]:
        for context_length in [256, 1024, 4096, 8192, 16384]:
            print("#" * 50)
            print(f"Running Experiments for d_model {d_model} and context_length {context_length}")
            data = torch.randn(batch_size, context_length, d_model, device=device)
            attn = VanillaAttention(d_model=d_model, max_seq_len=context_length, theta=10000, device=device)
            if compile_attn:
                attn = torch.compile(attn)
            try:
                for _ in range(warmup_steps):
                    output = attn(data)
                    output = output.sum()
                    output.backward()

                torch.cuda.synchronize()

                forward_times = []
                for _ in range(100):
                    start = timeit.default_timer()
                    attn(data)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    forward_times.append(end - start)

                print("Average forward pass time ", np.mean(forward_times))
                print("Std of forward pass time ", np.std(forward_times))

                torch.cuda.memory._record_memory_history(max_entries=1000000)
                output = attn(data)
                torch.cuda.memory._dump_snapshot("../outputs/memory_logs/vanilla_attention_snapshot.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
                print("forward pass memory recording finished")
                print("find memory file at: ", os.path.abspath("../outputs/memory_logs/vanilla_attention_snapshot.pickle"))

                backward_times = []
                for _ in range(100):
                    output = attn(data).sum()
                    torch.cuda.synchronize()
                    start = timeit.default_timer()
                    output.backward()
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    backward_times.append(end - start)

                print("Average backward pass time ", np.mean(backward_times))
                print("Std of backward pass time ", np.std(backward_times))


            except Exception as e:
                print(e)
            finally:
                print("Experiment Finished")

if __name__ == '__main__':
    profile_experiments(True)