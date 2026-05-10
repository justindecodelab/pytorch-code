import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 替换为你本地代理软件的端口，通常是 7890, 1080, 10809 等
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

# 1. preprocess the data

TRAIN_CONFIG = {
    'data_path': '../dataset',
    'epochs': 30,
    'learning_rate': 0.001,
    'batch_size': 500,
    'val_batch_size': 64,
    'test_batch_size': 64,
    'num_workers': 2,
    'val_ratio': 0.2,
    'model_path': 'model.pt',
}

# 转化为Tensor，将元素转化为0-1的数字，Normalize将其归一化。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATA_PATH = TRAIN_CONFIG['data_path']

# 训练集需要训练
full_trainset = torchvision.datasets.CIFAR10(
    DATA_PATH,
    train=True,
    transform=transform,
    target_transform=None,
    download=False,
)

train_size = int(len(full_trainset) * (1 - TRAIN_CONFIG['val_ratio']))
val_size = len(full_trainset) - train_size
trainset, valset = torch.utils.data.random_split(
    full_trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

# batch_size设置了批量大小，shuffle设置为True在装载过程中为随机乱序，num_workers>=1表示多线程读取数据。
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=TRAIN_CONFIG['batch_size'],
    shuffle=True,
    num_workers=TRAIN_CONFIG['num_workers'],
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=TRAIN_CONFIG['val_batch_size'],
    shuffle=False,
    num_workers=TRAIN_CONFIG['num_workers'],
)

# 测试集不需要训练
testset = torchvision.datasets.CIFAR10(
    DATA_PATH,
    train=False,
    transform=transform,
    target_transform=None,
    download=False,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=TRAIN_CONFIG['test_batch_size'],
    shuffle=False,
    num_workers=TRAIN_CONFIG['num_workers'],
)

# 指定类别标签
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# def imgshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# imgshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2. build simple CNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, padding=1, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)
 
    # 前向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # view将池化后的张量拉伸，-1的意思其实就是未知数的意思，根据其他位置（这里就是18*16*16）来推断这个-1是几
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()

# 3. loss function and optimizer
def createlossandoptimizer(net, learning_rate):
    loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)     # Adam 优化算法是随机梯度下降算法的扩展式
    print(optimizer)
    return loss, optimizer

# 4. define the training、evaluation and prediction functions
def get_train_loader(batch_size):
    # train_loader，一次以 batch_size 为一组循环读取训练数据
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
    )
    return train_loader


# 5. iterate the training and evaluation process
def trainNet(net, batchsize, n_epochs, learning_rate):
    print("HYPERPARAMETERS：")  
    print("batch-size=", batchsize)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)

    print("batchsize:", batchsize)
    train_loader = get_train_loader(batchsize)
    n_batches = len(train_loader)  # n_batches * batchsize = 20000（样本数目）
    print("n_batches", n_batches)
    loss, optimizer = createlossandoptimizer(net, learning_rate)
 
    training_start_time = time.time() 
    print("training start:")
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches 
        print("print_every:", print_every)  
        start_time = time.time()
        total_train_loss = 0
 
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # 将所有的梯度置零，防止每次 backward 的时候梯度累加
            optimizer.zero_grad() 
            # forward
            outputs = net(inputs)
            # loss
            loss_size = loss(outputs, labels)
            #backward
            loss_size.backward()
            # update weights
            optimizer.step()
            print(loss_size)
            loss_item = loss_size.item()
            running_loss += loss_item
            print("running_loss:", running_loss)
            total_train_loss += loss_item
            print("total_train_loss:", total_train_loss)
            # 在一个epoch里。每十组batchsize大小的数据输出一次结果，即以batch_size大小的数据为一组，到第10组，20组，30组...的时候输出
            if (i + 1) % 10 == 0:
                print("epoch{}, {:d} \t traing_loss:{:.2f} took:{:.2f}s".format(epoch + 1, int(100 * (i + 1) / n_batches),
                                                                                running_loss / 10, time.time()-
                                                                                start_time))
                running_loss = 0.0
                start_time = time.time()
 
        total_val_loss = 0
        
        for inputs, labels in val_loader:
            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

        # 验证集的平均损失
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))  

    # 所有的 Epoch 结束，也就是训练结束，计算花费的时间
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))  


def evaluate(net, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    trainNet(
        model,
        batchsize=TRAIN_CONFIG['batch_size'],
        n_epochs=TRAIN_CONFIG['epochs'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
    )
    torch.save(model.state_dict(), TRAIN_CONFIG['model_path'])
    model.load_state_dict(torch.load(TRAIN_CONFIG['model_path']))
    evaluate(model, testloader)
