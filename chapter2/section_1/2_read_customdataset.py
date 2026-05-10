import os
from PIL import Image
from torchvision import datasets
import torchvision.datasets.mnist as mnist

# 替换为你本地代理软件的端口，通常是 7890, 1080, 10809 等
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

# download MNIST dataset
train_set = datasets.MNIST('./dataset/MNIST', train=True, download=False)
test_set = datasets.MNIST('./dataset/MNIST', train=False, download=False)

# read files dataset
root = r'./dataset/MNIST/MNIST/raw'
train_set =(
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

# data num show
print('train set:', train_set[0].size())
print('test set:', test_set[0].size())

def convert_to_image(save_path, train=True):
    '''
    将图片存储在本地，并制作索引文件
    @para: save_path 图片存储路径，将在该路径下创建 train 和 test 两个文件夹
    @para: train 默认为True，本地存储训练集图片，否则存储测试集图片
    '''
    os.makedirs(save_path, exist_ok=True)

    if train:
        index_path = os.path.join(save_path, 'train.txt')
        data_path = os.path.join(save_path, 'train')
        images, labels = train_set
    else:
        index_path = os.path.join(save_path, 'test.txt')
        data_path = os.path.join(save_path, 'test')
        images, labels = test_set

    os.makedirs(data_path, exist_ok=True)

    with open(index_path, 'w') as file_obj:
        for i, (img, label) in enumerate(zip(images, labels)):
            img_path = os.path.join(data_path, f'{i}.jpg')
            Image.fromarray(img.numpy(), mode='L').save(img_path)
            file_obj.write(f'{i}.jpg {int(label)}\n')

# 根据需求本地存储训练集或测试集
save_path = './dataset/MNIST/mnist_data'
convert_to_image(save_path, train=True)
convert_to_image(save_path, train=False)


