import torch


a = torch.ones(3,3, requires_grad=True)
b = torch.ones(3,3, requires_grad=True)

c = a+b
z = c.sum()
z.backward()

print(a.grad)

z.brackward()
print(a.grad)