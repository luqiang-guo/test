import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(local_world_size, local_rank):

    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 0 uses GPUs [0, 1, 2, 3] and
    # rank 1 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )

    model = ToyModel().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2000, 10))
    labels = torch.randn(2000, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()


def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl")

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()

def test_mu(local_world_size, local_rank):
    # 新增：DDP backend初始化
    dist.init_process_group(backend="nccl")

    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    print("n = ", n)
    print("device_ids = ", device_ids)

    model = nn.Linear(10, 10).cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    # 是否需要把输入tensor放到GPU上 ???
    outputs = ddp_model(torch.randn(200, 10))
    labels = torch.randn(200, 10).to(device_ids[0])

    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # This is passed in via launch.py
    # parser.add_argument("--local_rank", type=int, default=0)
    # This needs to be explicitly passed in
    # parser.add_argument("--local_world_size", type=int, default=1)

    # args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    # spmd_main(args.local_world_size, args.local_rank)

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    
    test_mu(int(env_dict["WORLD_SIZE"]), int(env_dict["RANK"]))


# python -m torch.distributed.launch --nproc_per_node 2 ddp_train_torch_mu.py --local_world_size=2


# MASTER_ADDR=127.0.0.1 MASTER_PORT=12345 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0  python ddp_train_torch_mu.py
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12345 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 python ddp_train_torch_mu.py

# OMP_NUM_THREADS=1  MASTER_ADDR=127.0.0.1 MASTER_PORT=12345 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 /usr/local/cuda/bin/nsys profile --force-overwrite=true -o ddp0.qdrep python ddp_train_torch_mu.py
# MASTER_ADDR=127.0.0.1 MASTER_PORT=12345 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1 /usr/local/cuda/bin/nsys profile --force-overwrite=true -o ddp1.qdrep python ddp_train_torch_mu.py
