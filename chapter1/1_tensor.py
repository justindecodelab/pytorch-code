import torch
x = torch.empty(5, 3)
print(type(x))
print(x)

tensor_ones = torch.ones(2, 3)
print(tensor_ones)
tensor_zeros = torch.zeros(2, 3)
print(tensor_zeros)

print(tensor_zeros.dtype)
tensor_zeros_int = torch.zeros(2, 3, dtype=torch.long)
print(tensor_zeros_int.dtype)

x = torch.rand(5, 3)
print(x)

x = torch.randn(4, 4)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = torch.tensor([5.5, 3], dtype=torch.double)
print(x.dtype)
print(x.shape)
x = x.new_ones(5, 3)
print(x)
print(x.shape)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# in function: shape == size()

x_size = x.size()
print(x_size)
row, col = x_size
print(row, col)

x_range = torch.arange(1, 10, 1)
print(x_range)

