import torch
print(torch.__version__)

a = torch.rand(3,3)
b = torch.rand(3,3)

print(a)
print(b)

c = torch.matmul(a, b)
print(c)