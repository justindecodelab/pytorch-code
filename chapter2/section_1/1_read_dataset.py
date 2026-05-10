import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 替换为你本地代理软件的端口，通常是 7890, 1080, 10809 等
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

# read example 1 (auto download from online)
dataset_dir = '../dataset/'
train_data = torchvision.datasets.CIFAR10(
    dataset_dir,
    train=True,
    transform=None,
    target_transform=None,
    download=False,
)
test_data = torchvision.datasets.CIFAR10(
    dataset_dir,
    train=False,
    transform=None,
    target_transform=None,
    download=False,
)

# read example 2 (example 1 with data augmentation)
custom_transform = transforms.transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
])

train_data = torchvision.datasets.CIFAR10('../dataset',
            train = True,
            transform = custom_transform,
            target_transform = None,
            download = False
)

# DataLoader usage
train_loader = DataLoader(train_data,
                          batch_size = 2,
                          shuffle = True,
                          num_workers = 4)

if __name__ == '__main__':
    train_loader = DataLoader(
        train_data,
        batch_size=32,      # 实际训练建议设为 32， 64 或 128
        shuffle = True,     # 训练集通常需要打乱
        num_workers = 8     # num_workers >= 1 表示多进程加载数据
    )

    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"tensor shape: {images.shape}")
    print(f"labels: {labels}")
