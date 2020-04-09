import torch
import numpy as np

# arr = np.zeros((5,5))
# np.put(arr, [0, 1, 2, 9], 1)
# print(arr)

# tensor = torch.zeros((5,5), dtype=torch.float)
# tensor[0][1] = 1.0
# tensor[2][3] = 1.0
# indices = (tensor == 1.0).nonzero()
# print(tensor)
# print(indices)
# mask = torch.zeros_like(tensor, dtype=torch.float)
# mask.scatter_(0, tensor.type(torch.long), 2.0)
# print(mask)

output = torch.zeros((10,5), dtype=torch.float)
output[0][1] = 1.0
output[2][3] = 1.0
print(output)
ids = torch.argmax(output).numpy()
print(ids)
indices = (output == 1.0).nonzero()
indices = [index[0] * output.shape[1] + index[1] for index in indices]
print(indices)
mask = np.zeros(output.shape)
np.put(mask, indices, 1)
mask = torch.FloatTensor(mask)
print(mask)