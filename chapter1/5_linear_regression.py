# Simple definition of linear regression: assuming the dependent variable has a linear relationship with respect to the independent variable.
# for example: y = wx +b

import torch
import numpy as np
import matplotlib.pylab as plt

# process 1: import artificial data
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
[9.779], [6.182], [7.59], [2.167], [7.042], 
[10.791], [5.313], [7.997], [3.1]], dtype=np.float32) 
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
[3.366], [2.596], [2.53], [1.221], [2.827], 
[3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# process 2: transfer to Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo')
# plt.show()

# process 3: build model
# define the linear model
# y = w * x + b
w = torch.tensor([-1.], requires_grad=True) 
b = torch.tensor([0.], requires_grad=True) 
def linear_model(x):
    return x * w + b

# plot estimate result before train
# y_ = linear_model(x_train)
# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo',label='real')
# plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro',label='estimate')
# plt.legend() # 图例 It is usually used in conjunction with the label parameter
# plt.show()

# process 4: define loss function
# tell pytorch how to optimize our model
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2) # least squares method

# process 5: training iterations using gradient descent
# train 10 iteration
lr = 1e-2
for e in range(10):
    y_ = linear_model(x_train)

    # compute loss
    loss = get_loss(y_, y_train)
    loss.backward()

    # manually update parameters
    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data
    print('epoch: {}, loss: {}'.format(e, loss))

    # reset the grad to zero
    w.grad.zero_()
    b.grad.zero_()

# plot estimate result after train
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimate')
plt.legend()
plt.show()

