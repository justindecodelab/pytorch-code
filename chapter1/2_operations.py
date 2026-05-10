import torch

x = torch.ones(5, 3)
print(x)

# plus syntax: 
y = torch.rand(5, 3)
z1 = x + y
print(z1)

z2 = torch.add(x, y)
print(z2)

z3 = torch.empty(5, 3)
torch.add(x, y, out=z3) # plus value equal z3
print(z3)

y.add_(x)
print(y)

a = torch.randn(2, 3)
print(a)
b = torch.randn(2,3)
print(b)

c = torch.sub(a, b)
print(c)
d = torch.mul(a, b)
print(d)
e = torch.div(a, b)
print(e)

a = torch.randn(2, 3)
print(a)
b = torch.randn(3, 2)
print(b)
c = torch.mm(a, b)
print(c)

a = torch.randn(2, 3)
print(a)
b = torch.abs(a)
print(b)

b = torch.pow(a, 2)
print(b)

x = torch.randn(3, 3)
print(x)
print(x[:, 0])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 2) # 使用-1时pytorch将会自动根据其他维度进行推导
print(x.size(), y.size(), z.size())

x = torch.randn(2, 3)
print(x)
x_cat_dim0 = torch.cat((x,x), dim=0)
x_cat_dim1 = torch.cat((x,x,x), dim=1)
print(x_cat_dim0)
print(x_cat_dim1)

# 3 dim
x = torch.randn(2, 3, 4) # 表示2个3 * 4的特征矩阵
print(x)

x = torch.randn(1)
print(x)
print(x.size()) # size() == shape
print(x.item())
print(type(x.item()))

a = torch.randn(2, 3)
print(a)
b = torch.clamp(a, -0.5, 0.5)
print(b)




