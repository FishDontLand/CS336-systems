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

def naive_dist_training(rank: int, world_size: int, warmup_steps: int, model: torch.nn.Module, data: torch.tensor,
                  device_type: str, shared_dict=None, tracking_op=None, use_flatten=False):
    if tracking_op is not None and tracking_op not in ['full_step', 'communication']:
        raise ValueError('Unknown tracking operation')
    setup(rank, world_size, device_type)
    batch_size = data.size(0) // world_size
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    data_batch = data[start_idx:end_idx]
    data_batch = data_batch.to(device_type)
    assert len(data_batch) == batch_size

    rank_model = copy.deepcopy(model)
    rank_model = rank_model.to(device_type)
    optimizer = AdamW(rank_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    all_run_time = [None for _ in range(world_size)]
    for i in range(warmup_steps + 1):
        if i == warmup_steps and tracking_op == 'full_step':
            dist.barrier()
            start_time = timeit.default_timer()
        optimizer.zero_grad()
        output = rank_model(data_batch)
        loss = (output * output).mean()
        loss.backward()

        if i == warmup_steps and tracking_op == 'communication':
            dist.barrier()
            start_time = timeit.default_timer()
        if use_flatten:
            flattened_grads = []
            for p in rank_model.parameters():
                if p.grad is not None:
                    flattened_grads.append(p.grad.flatten())
            flattened_grads = torch.concat(flattened_grads)
            if device_type == 'cuda':
                dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
            else:
                dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=False)
        else:
            for p in rank_model.parameters():
                if p.grad is not None:
                    if device_type == 'cuda':
                        dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.AVG, async_op=False)
                    else:
                        dist.all_reduce(tensor=p.grad, op=dist.ReduceOp.SUM, async_op=False)

        if i == warmup_steps and tracking_op == 'communication':
            dist.barrier()
            end_time = timeit.default_timer()
            dist.all_gather_object(all_run_time, end_time - start_time)

        if use_flatten:
            if device_type != 'cuda':
                flattened_grads = flattened_grads / world_size
            offset = 0
            for p in rank_model.parameters():
                if p.grad is not None:
                    p.grad = flattened_grads[offset : (offset + p.grad.numel())].view(p.size())
                    offset += p.grad.numel()
        else:
            if device_type != 'cuda':
                for p in rank_model.parameters():
                    if p.grad is not None:
                        p.grad = p.grad / world_size

        optimizer.step()
        if i == warmup_steps and tracking_op == 'full_step':
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            end_time = timeit.default_timer()
            dist.all_gather_object(all_run_time, end_time - start_time)

    if rank == 0:
        if shared_dict is not None:
            shared_dict['result'] = rank_model.to('cpu').state_dict()
        if tracking_op is not None:
            avg_time = sum(all_run_time) / len(all_run_time)
            print(f'average run time: {avg_time}')

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
    mp.spawn(fn=naive_dist_training, args=(world_size, train_steps - 1, model, data, device_type, shared_dict, None, True), nprocs=world_size, join=True)


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
        self.mb_to_bytes = 2 ** 20

        self.construct_buckets(self.module.named_parameters())

        for tensor in self.bucket_tensors:
            self.handles.append(dist.broadcast(tensor.data, 0, async_op=True))

        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for name, p in self.module.named_parameters():
            restored_param = torch.empty(p.numel(), dtype=p.dtype, device=p.device)
            offset = 0
            for bucket_idx, start, end in self.linked_bucket[name]:
                restored_param[offset:offset + (end - start)] = self.bucket_tensors[bucket_idx][start:end]
                offset += end - start
            p.data = restored_param.view(p.size())

        # only keep track of the parameters with gradients
        self.construct_buckets(self.get_grad_param_iter())

        for name, p in self.get_grad_param_iter():
            p.register_post_accumulate_grad_hook(self.hook)

    def get_grad_param_iter(self)->Iterator[tuple[str, Parameter]]:
        named_param_lst = list(self.module.named_parameters())
        for i in range(len(named_param_lst) - 1, -1, -1):
            n, p = named_param_lst[i]
            if p.requires_grad:
                yield n, p

    def construct_buckets(self, param_iter: Iterator[tuple[str, Parameter]]):
        self.linked_bucket = dict()
        self.bucket_tensors = []
        self.bucket_param_count = []
        self.counter = []
        self.param_to_name = dict()

        bucket = []
        bucket_names = []
        bucket_offsets = [0]
        size = 0
        bucket_idx = 0

        for name, p in param_iter:
            if self.is_cuda is None:
                self.is_cuda = p.is_cuda

            self.param_to_name[id(p)] = name
            current_p = p.flatten()
            bucketed_elements = 0

            while bucketed_elements < p.numel():
                remain_size = (p.numel() - bucketed_elements) * p.element_size() / self.mb_to_bytes
                if size + remain_size > self.bucket_size_mb:
                    ratio = (self.bucket_size_mb - size) / remain_size
                    bucket_remain_elements = int(p.numel() * ratio)

                    if bucket_remain_elements > 0:
                        bucket.append(current_p[bucketed_elements:bucketed_elements + bucket_remain_elements])
                        bucket_names.append(name)
                        bucket_offsets.append(bucket_offsets[-1] + bucket_remain_elements)

                    num_params = len(bucket)
                    bucket_tensor = torch.concat(bucket)
                    bucketed_elements += bucket_remain_elements

                    for i in range(len(bucket_names)):
                        if bucket_names[i] not in self.linked_bucket:
                            self.linked_bucket[bucket_names[i]] = [[bucket_idx, bucket_offsets[i], bucket_offsets[i + 1]]]
                        else:
                            self.linked_bucket[bucket_names[i]].append([bucket_idx, bucket_offsets[i], bucket_offsets[i + 1]])

                    self.bucket_tensors.append(bucket_tensor)
                    self.bucket_param_count.append(num_params)
                    self.counter.append(0)

                    size = 0
                    bucket_idx += 1
                    bucket_names.clear()
                    bucket.clear()
                    bucket_offsets.clear()
                    bucket_offsets.append(0)
                else:
                    bucket.append(current_p[bucketed_elements:])
                    bucket_names.append(name)
                    size += (p.numel() - bucketed_elements) * p.element_size() / self.mb_to_bytes
                    bucket_offsets.append(bucket_offsets[-1] + p.numel() - bucketed_elements)
                    bucketed_elements = p.numel()

        if bucket:
            num_params = len(bucket)
            bucket_tensor = torch.concat(bucket)
            self.handles.append(dist.broadcast(bucket_tensor, 0, async_op=True))
            for i in range(len(bucket_names)):
                if bucket_names[i] not in self.linked_bucket:
                    self.linked_bucket[bucket_names[i]] = [[bucket_idx, bucket_offsets[i], bucket_offsets[i + 1]]]
                else:
                    self.linked_bucket[bucket_names[i]].append([bucket_idx, bucket_offsets[i], bucket_offsets[i + 1]])

            self.bucket_tensors.append(bucket_tensor)
            self.bucket_param_count.append(num_params)
            self.counter.append(0)

    def hook(self, p: torch.tensor):
        offset = 0
        flat_grad = p.grad.flatten()
        name = self.param_to_name[id(p)]
        for bucket_idx, start, end in self.linked_bucket[name]:
            self.bucket_tensors[bucket_idx][start:end] = flat_grad[offset:offset + end - start]
            self.counter[bucket_idx] += 1
            offset += end - start

            if self.counter[bucket_idx] == self.bucket_param_count[bucket_idx]:

                if self.is_cuda:
                    self.handles.append(dist.all_reduce(self.bucket_tensors[bucket_idx], op=dist.ReduceOp.AVG, async_op=True))
                else:
                    self.handles.append(dist.all_reduce(self.bucket_tensors[bucket_idx], op=dist.ReduceOp.SUM, async_op=True))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for name, p in self.get_grad_param_iter():
            restored_param = torch.empty(p.numel(), device=p.device, dtype=p.grad.dtype)
            offset = 0
            for bucket_idx, start, end in self.linked_bucket[name]:
                restored_param[offset:offset + end - start] = self.bucket_tensors[bucket_idx][start:end]
                offset += end - start

            p.grad = restored_param.view(p.size())
            if not self.is_cuda:
                p.grad /= dist.get_world_size()

if __name__ == '__main__':
    pass

