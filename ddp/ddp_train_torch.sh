python -m torch.distributed.launch --nproc_per_node 2 ddp_train_torch.py --local_world_size=2

# /usr/local/cuda/bin/nsys profile --force-overwrite=true --trace-fork-before-exec=true --stats=false -o ddp.qdrep sh ddp_train_torch.sh
