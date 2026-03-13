import timeit
from typing import Iterator

import torch
import torch.distributed as dist
import copy
from cs336_basics.module import Linear, RMSNorm
import torch.multiprocessing as mp
import os
from cs336_basics.optimizer import AdamW
from torch.nn import Parameter


def setup(rank: int, world_size: int, device_type: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if device_type == 'cuda':
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype|None):
        super().__init__()
        self.fc1 = Linear(in_features, 10, dtype=dtype)
        self.rms_norm = RMSNorm(10, dtype=dtype)
        self.fc2 = Linear(10, out_features, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.rms_norm(x)
        x = self.fc2(x)
        return x

def non_parallel_train(train_steps, model: torch.nn.Module, data:torch.tensor, device_type: str):
    non_parallel_model = copy.deepcopy(model)
    non_parallel_model = non_parallel_model.to(device_type)
    optimizer = AdamW(non_parallel_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

    for _ in range(train_steps):
        optimizer.zero_grad()
        output = non_parallel_model(data)
        loss = (output * output).mean()
        loss.backward()
        optimizer.step()

    return non_parallel_model.to('cpu').state_dict()

def compare_training_result(device_type='cpu', train_steps=1):
    world_size = 4
    batch_size = 64
    seq_len = 1024
    feature_size = 128
    data = torch.randn(batch_size, seq_len, feature_size)

    if device_type == 'cuda' and torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device_type = 'cuda'
    elif device_type != 'cpu':
        print(f'device type {device_type} not supported, switched to cpu mode')
        device_type = 'cpu'

    model = ToyModel(feature_size, feature_size, dtype=torch.float32)
    non_parallel_result = non_parallel_train(train_steps, model, data, device_type)

    manager = mp.Manager()
    shared_dict = manager.dict()
    mp.spawn(fn=dist_training, args=(world_size, train_steps - 1, model, data, device_type, shared_dict, None, True), nprocs=world_size, join=True)


    max_abs_diff = 0
    for name in non_parallel_result:
        param1 = non_parallel_result[name]
        param2 = shared_dict['result'][name]
        max_abs_diff = max(max_abs_diff, torch.abs(param1 - param2).max())
        assert torch.allclose(param1, param2), torch.abs(param1 - param2).max()
    print(max_abs_diff)


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.is_cuda = None
        self.handles = []
        for p in self.module.parameters():
            if self.is_cuda is None:
                self.is_cuda = p.is_cuda
            self.handles.append(dist.broadcast(p.data, 0, async_op=True))
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.hook)
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def hook(self, tensor: torch.tensor) -> None:
        if self.is_cuda:
            handle = dist.all_reduce(tensor.grad, op=dist.ReduceOp.AVG, async_op=True)
        else:
            handle = dist.all_reduce(tensor.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        if not self.is_cuda:
            group_size = dist.get_world_size()
            for p in self.module.parameters():
                if p.grad is not None:
                    p.grad = p.grad / group_size

class BucketDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.handles = []
        self.is_cuda = None

        self.construct_buckets(self.module.named_parameters())
        for tensor in self.bucket_tensors:
            self.handles.append(dist.broadcast(tensor.data, 0, async_op=True))

        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for name, p in self.module.named_parameters():
            bucket_idx, offset = self.linked_bucket[name]
            p.data = self.bucket_tensors[bucket_idx][offset:offset + p.numel()].view(p.size())

        # only keep track of the parameters with gradients
        self.construct_buckets(self.get_grad_param_iter())

        for name, p in self.get_grad_param_iter():
            p.register_post_accumulate_grad_hook(lambda x : self.hook(x.grad, name))

    def get_grad_param_iter(self)->Iterator[tuple[str, Parameter]]:
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                yield n, p

    def construct_buckets(self, param_iter: Iterator[tuple[str, Parameter]]):
        self.linked_bucket = dict()
        self.bucket_tensors = []
        self.bucket_param_count = []
        self.counter = []

        bucket = []
        bucket_names = []
        bucket_offsets = []
        size = 0
        bucket_idx = 0

        for name, p in param_iter:
            if self.is_cuda is None:
                self.is_cuda = p.is_cuda

            param_size = p.numel() * p.element_size()
            if param_size / 1e6 > self.bucket_size_mb:
                raise ValueError("Bucket Size too small!")
            if size + param_size > self.bucket_size_mb:
                num_params = len(bucket)
                bucket_tensor = torch.concat(bucket)
                self.handles.append(dist.broadcast(bucket_tensor, 0, async_op=True))
                for i in range(len(bucket_names)):
                    self.linked_bucket[bucket_names[i]] = [bucket_idx, bucket_offsets[i]]
                self.bucket_tensors.append(bucket_tensor)
                self.bucket_param_count.append(num_params)
                self.counters.append(0)

                size = 0
                bucket_idx += 1
                bucket_names.clear()
                bucket.clear()
                bucket_offsets.clear()
                bucket_offsets.append(0)

            bucket.append(p.flatten())
            bucket_names.append(name)
            size += param_size
            bucket_offsets.append(bucket_offsets[-1] + p.numel())

        if bucket:
            num_params = len(bucket)
            bucket_tensor = torch.concat(bucket)
            self.handles.append(dist.broadcast(bucket_tensor, 0, async_op=True))
            for i in range(len(bucket_names)):
                self.linked_bucket[bucket_names[i]] = [bucket_idx, bucket_offsets[i]]
            self.bucket_tensors.append(bucket_tensor)
            self.bucket_param_count.append(num_params)
            self.counters.append(0)

    def hook(self, p: torch.tenor, name: str):
        hooked_bucket_idx, offset = self.linked_bucket[name]
        self.bucket_tensors[hooked_bucket_idx][offset : offset + p.numel()] = p.grad.flatten()
        self.counter[hooked_bucket_idx] += 1
        if self.counter[hooked_bucket_idx] == self.bucket_param_count[hooked_bucket_idx]:
            if self.is_cuda:
                self.handles.append(dist.all_reduce(self.bucket_tensors[hooked_bucket_idx], op=dist.ReduceOp.AVG, async_op=True))
            else:
                self.handles.append(dist.all_reduce(self.bucket_tensors[hooked_bucket_idx], op=dist.ReduceOp.SUM, async_op=True))

    def forward(self, *inputs, **kwargs):
        for i in range(len(self.bucket_tensors)):
            self.bucket_tensors[i] = torch.empty(self.bucket_tensors[i].size(), device=self.bucket_tensors[i].device)
            self.counters[i] = 0
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for name, p in self.get_grad_param_iter():
            link_idx, offset = self.linked_bucket[name]
            p.grad = self.bucket_tensors[link_idx][offset:offset + p.numel()].view(p.size())
            if not self.is_cuda:
                p.grad /= dist.get_world_size()

if __name__ == '__main__':
    pass

