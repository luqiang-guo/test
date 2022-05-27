import torch


a = torch.tensor([1,2,3], dtype=torch.int32)
b = torch.tensor([1,2,3],dtype=torch.int32)


print(torch.add(a, b))