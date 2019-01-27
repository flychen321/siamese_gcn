import numpy as np
import torch
import scipy.sparse as sp

# row = np.array([0, 3, 1, 0])
# col = np.array([0, 3, 1, 2])
# data = np.array([4, 5, 7, 9])
# a = sp.coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
# print(a)
#
# indptr = np.array([0, 3, 4, 7])
# indices = np.array([0, 1, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6, 7])
# b = sp.csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
# print(b)
#
# indptr = np.array([0, 3, 4, 7])
# indices = np.array([0, 1, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6, 7])
# c = sp.csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
# print(c)

a = np.random.rand(4,4)
b = np.random.rand(4,4)
print(a)
print(b)
print(a.dot(b))

a = torch.from_numpy(a)
b = torch.from_numpy(b)
print(a)
print(b)
print(a.mm(b))