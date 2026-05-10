import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class MnistDataset(Dataset):

    def __init__(self, image_path, image_label, transform=None):
        super(MnistDataset, self).__init__()
        self.image_path = image_path    # 初始化图像路径列表
        self.image_label = image_label  # 初始化图像标签列表
        self.transform = transform      # 初始化数据增强方法

    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """

        image = Image.open(self.image_path[index])  # 读取图像
        image = np.array(image)                     # 转换为numpy数组
        label = float(self.image_label[index])      # 获取对应标签

        if self.transform is not None:
            image = self.transform(image)            # 对图像进行数据增强

        return image, torch.tensor(label)           # 返回图像和标签，标签转换为torch.tensor类型
    
    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.image_path)                 # 数据集大小等于图像路径列表的长度
    
def get_path_label(img_root, label_file_path):
    """
    获取数字图像的路径和标签并返回对应的列表
    @para: img_root 图像存储的根目录  
    @para: label_file_path 保存图像标签数据的文件路径 .csv 或 .txt 分隔符为','
    @return: image_paths 图像路径列表和labels标签列表
    """
    data = pd.read_csv(label_file_path, sep=r'\s+', names=['img', 'label'])
    data['img'] = data['img'].apply(lambda x: os.path.join(img_root, x))
    return data['img'].tolist(), data['label'].tolist()

# 获取训练集路径列表和标签列表
train_data_root = './dataset/MNIST/mnist_data/train/'
train_label = './dataset/MNIST/mnist_data/train.txt'
train_img_list, train_label_list = get_path_label(train_data_root, train_label)  
# 训练集dataset
train_dataset = MnistDataset(train_img_list,
                             train_label_list,
                             transform=transforms.Compose([transforms.ToTensor()]))

# 获取测试集路径列表和标签列表
test_data_root = './dataset/MNIST/mnist_data/test/'
test_label = './dataset/MNIST/mnist_data/test.txt'
test_img_list, test_label_list = get_path_label(test_data_root, test_label)
# 测试集dataset
test_dataset = MnistDataset(test_img_list,
                            test_label_list,
                            transform=transforms.Compose([transforms.ToTensor()]))

# 使用DataLoader批量读取数据
# 训练数据加载
train_loader = DataLoader(dataset=train_dataset,  # 加载的数据集（Dataset对象）
                         batch_size=3,  # 一个批量大小
                         shuffle=True,  # 是否打乱数据顺序
                         num_workers=4)  # 使用多进程加载的进程数，0代表不使用多进程（win系统建议改成0）
# 测试数据加载
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=3,
                        shuffle=False,
                        num_workers=4)


if __name__ == '__main__':
    # 直接使用Dataset对象创建迭代器
    # train_iter = iter(train_dataset)
    # test_iter = iter(test_dataset)

    # 使用DataLoader创建迭代器
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    print('len(train_dataset):', len(train_dataset))
    print('next(iter(train_dataset)):')
    print(next(train_iter))
    print('-' * 50)

    print('len(test_dataset):', len(test_dataset))
    print('next(iter(test_dataset)):')
    print(next(test_iter))

    # for i in train_dataset:
    #     img, label = i
    #     print(img.size(), label)

    for i, img_data in enumerate(train_loader, 1):
        images, labels = img_data
        # batch1:images shape info-->torch.Size([3, 1, 28, 28]) labels-->tensor([5., 0., 4.]) 每（batch_size）个批量的图像张量形状和对应标签，组成一个二元元组
        print('batch{0}:images shape info-->{1} labels-->{2}'.format(i, images.shape, labels)) 