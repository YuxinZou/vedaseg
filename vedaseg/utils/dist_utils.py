# modify from mmcv and mmdetection

import os

import numpy as np
import torch
import torch.distributed as dist


def init_dist_pytorch(backend='nccl', **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def reduce_tensor(data, average=True):
    rank, world_size = get_dist_info()
    if world_size < 2:
        return data

    with torch.no_grad():
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).cuda()
        dist.reduce(data, dst=0)
        if rank == 0 and average:
            data /= world_size
    return data


def gather_tensor(data):
    rank, world_size = get_dist_info()
    if world_size < 2:
        return data, data.shape[0]

    with torch.no_grad():
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).cuda()

        # gather all result part tensor shape
        shape_tensor = torch.tensor(data.shape).cuda()
        shape_list = [shape_tensor.clone() for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)
        # padding result part tensor to max length
        max_size_index = np.argmax([size[0] for size in shape_list])
        shape_max = tuple(shape_list[max_size_index])
        part_send = torch.ones(shape_max, dtype=data.dtype).cuda() * 255
        part_send[:shape_tensor[0]] = data
        data_list = [data.new_zeros(shape_max) for _ in range(world_size)]
        # gather all result part
        dist.all_gather(data_list, part_send)
        gather_data = torch.cat(data_list, 0)

    return gather_data, shape_max[0].item()



def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
