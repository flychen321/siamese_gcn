import numpy as np
import torch

a = [['a', 1], ['b', 2], ['c', 3]]
# a = [[5, 1], [6, 2], [7, 3]]

b = torch.FloatTensor(3, 5).zero_()

b[0] = torch.Tensor(5)
print(torch.Tensor((9,6,7)))
print(b[0])
print(b)