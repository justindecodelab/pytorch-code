import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import matplotlib.pyplot as plt

# process 1: Building a fully connected network
# define network structrue
class Net(nn.Module):
    def __init__(self, in_c=784, out_c=10):
        super(Net, self).__init__()

        # define full-connected layer
        self.fc1 = nn.Linear(in_c, 512)
        # define activate layer
        self.act1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, out_c)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)

        return x
    
# build network
net = Net()

# process 2: Data loading and network input
# preparing dataset
# traning set
train_set = mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# test set
test_set = mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

# training set loader
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
# test set loader
test_data = DataLoader(train_set, batch_size=128, shuffle=False)

# Visualize data
# import random
# for i in range(4):
#     ax = plt.subplot(2, 2, i+1)
#     idx = random.randint(0, len(train_set))
#     digit_0 = train_set[idx][0].numpy()
#     digit_0_image = digit_0.reshape(28, 28)
#     ax.imshow(digit_0_image, interpolation="nearest")
#     ax.set_title('label: {}'.format(train_set[idx][1]))
# plt.show()
    

# process 3: Define the loss function and optimizer
# define loss function -- CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# define optimizer --SGD stochastic gradient descent
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)

# process 4: Starting train
# recording Training Loss
losses = []
# recording Training Accuracy
acces = []
# recording Test Loss
eval_losses = []
# ​recording Test Accuracy
eval_acces = []
# set the number of iterations
nums_epoch = 20
for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for batch, (img, label) in enumerate(train_data):
        # batch 是索引，img 是数据，label 是标签
        img = img.reshape(img.size(0), -1)
        img = Variable(img) # 新写法：img = img.to(device)
        label = Variable(label)

        # forward propagation 
        out = net(img)                  # 1. predict model
        loss = criterion(out, label)    # 2. calculate loss
        # backward propagation
        optimizer.zero_grad()   # 3. clear gradient
        loss.backward()         # 4. calculate gradient
        optimizer.step()        # 5. update parameter

        # recording error
        train_loss += loss.item()
        # calculate the accuracy of classfication
        _, pred = out.max(1)                        # 获取预测类别 shape [batch_size, num_classes] 其中_：表示忽略索引0，索引0：表示out shape的batch_size(批次维度)，索引1：表示out shape的num_class(类别维度)
        num_correct = (pred == label).sum().item()  # 统计正确数
        acc = num_correct / img.shape[0]            # 计算准确率

        if (batch + 1) % 200 == 0:
             print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                 batch+1,
                                                                                 loss.item(),
                                                                                 acc))
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data)) 

    eval_loss = 0
    eval_acc = 0

    # Test set not trained
    for img, label in test_data:
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        out = net(img)
        loss = criterion(out, label)
        # recording error
        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data))) 


plt.figure()
plt.suptitle('Test', fontsize=12)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(eval_losses, color='r')
ax1.plot(losses, color='b')
ax1.set_title('Loss', fontsize=10, color='black')
ax2 = plt.subplot(1, 2, 2)
ax2.plot(eval_acces, color='r')
ax2.plot(acces, color='b')
ax2.set_title('Acc', fontsize=10, color='black')
plt.show()

