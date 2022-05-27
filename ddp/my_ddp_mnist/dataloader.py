import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def init_dataloader(batch_size, use_cuda):
    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    return train_loader, test_loader

def init_ddp_dataloader(batch_size, use_cuda):
    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('./data', train=False,
                       transform=transform)

    train_data_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    test_data_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=local_batch_size,
                                        sampler=train_data_sampler,
                                        pin_memory=True, num_workers=num_data_processes,
                                        collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=local_batch_size,
                                        sampler=test_data_sampler,
                                        pin_memory=True, num_workers=num_data_processes,
                                        collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    # train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    return train_loader, test_loader