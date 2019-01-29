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
# a = np.arange(16).reshape(4,4)
# print(a)
# for i in range(len(a)):
#     print(i)
#     b = a[:i]
#     c = a[i:]
#     d = np.concatenate((c,b), 0)
#     print(b)
#     print(c)
#     print(d)

# a = torch.arange(16).reshape(4,4)
# print(a)
# for i in range(len(a)):
#     print(i)
#     b = a[:i]
#     c = a[i:]
#     d = torch.cat((c,b), 0)
#     print(b)
#     print(c)
#     print(d)

a = np.ones((3, 5))*5
b = np.ones((5,))*4
c = np.subtract(a, b)
print(c)