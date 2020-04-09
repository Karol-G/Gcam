import torch
import numpy as np

a = torch.randn(4, 1)
b = torch.randn(1, 4)
c = torch.mul(a, b)
d = np.matmul(a.detach().cpu().numpy(), b.detach().cpu().numpy())
e = np.multiply(a.detach().cpu().numpy(), b.detach().cpu().numpy())
print(c)
print(d)
print(e)