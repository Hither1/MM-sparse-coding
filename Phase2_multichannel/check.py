import torch
import torch.nn as nn

a = nn.Parameter(torch.zeros(5))
b = (a**2).sum()
b.backward()
print(a.grad)