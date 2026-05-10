# general process of backpropagation
import torch, torchvision
model = torchvision.models.resnet18(pretrained=False)
data = torch.rand(1, 3, 64, 54)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()
loss.backward() # backward

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() # gradient descent

# simple experimental test to achieve automatic gradient differentiation
x1 = torch.ones((2, 2))
print(x1)
print(x1.grad_fn)

x2 = torch.ones((2, 2), requires_grad=True)
print(x2)
print(x2.grad_fn)

print(x1.is_leaf, x2.is_leaf)

y2 = x2 + 1
print(y2.requires_grad)

a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)
y = a**2 + 2*b**3
print(y)

y.backward() # backward function 计算 ∂y/∂x gradient(梯度) == slope(斜率) 
print(a.grad)
print(b.grad)
