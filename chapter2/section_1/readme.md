# Chapter 2 程序总结

本章核心主题是图像数据读取、数据集封装、分类目录组织，以及常见图像变换的使用。

从代码结构上看，这一章大致分为 4 个部分：

1. 使用 `torchvision` 直接读取标准数据集
2. 将原始数据导出成本地图片和标签文件
3. 自定义 `Dataset` 与 `DataLoader`
4. 使用 `ImageFolder` 和 `transforms` 处理分类图像数据

下面按程序逐个总结。

## 1_read_dataset.py

### 程序作用

这个脚本演示了如何直接使用 `torchvision.datasets.CIFAR10` 读取 CIFAR10 数据集，并配合 `DataLoader` 生成批量数据。

### 关键流程

1. 设置代理环境变量
2. 读取 CIFAR10 训练集和测试集
3. 定义图像增强 `custom_transform`
4. 构建 `DataLoader`
5. 取出一个 batch 并打印形状与标签

### 关键函数与参数说明

#### `torchvision.datasets.CIFAR10(...)`

示例参数：

```python
torchvision.datasets.CIFAR10(
		dataset_dir,
		train=True,
		transform=None,
		target_transform=None,
		download=False,
)
```

参数说明：

- `dataset_dir`
	数据集根目录。
- `train=True`
	`True` 表示训练集，`False` 表示测试集。
- `transform=None`
	图像预处理或增强方法，作用在输入图像上。
- `target_transform=None`
	标签变换，通常分类任务里保持 `None` 即可。
- `download=False`
	是否自动下载。第一次运行且本地没有数据集时应改为 `True`。

参数为什么这样选：

- 训练集必须用 `train=True`，测试集必须用 `train=False`，否则训练和评估数据会混淆。
- `download=False` 适合已经提前下载好的环境；教学脚本里这样写可以避免每次运行都重复联网检查。

#### `transforms.Compose([...])`

示例：

```python
transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ColorJitter(0.2, 0.2, 0.2),
		transforms.RandomRotation(5),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
												 [0.229, 0.224, 0.225])
])
```

参数说明与选值解释：

- `Resize((64, 64))`
	将原图统一缩放到 `64 x 64`。
	这样选是为了让后续网络输入尺寸固定。
- `ColorJitter(0.2, 0.2, 0.2)`
	分别控制亮度、对比度、饱和度扰动强度。
	`0.2` 属于较温和的增强，不会过度破坏原图语义。
- `RandomRotation(5)`
	在 `[-5, 5]` 度内随机旋转。
	5 度通常用于轻量几何增强，适合 CIFAR 这种小图。
- `ToTensor()`
	将 `PIL.Image` 转为 `Tensor`，并把像素从 `[0, 255]` 归一到 `[0, 1]`。
- `Normalize(mean, std)`
	按通道做标准化。
	这里使用的是 ImageNet 常见均值和标准差，是迁移学习场景里常见的默认值。

#### `DataLoader(...)`

示例：

```python
DataLoader(
		train_data,
		batch_size=32,
		shuffle=True,
		num_workers=8,
)
```

参数说明：

- `batch_size=32`
	每次返回 32 张图像。32 是显存占用和训练稳定性之间常见的平衡点。
- `shuffle=True`
	训练集通常需要打乱，避免数据顺序对梯度更新造成偏差。
- `num_workers=8`
	用 8 个子进程并行加载数据，提升读取效率。
	如果是 Windows 或资源较少环境，通常会改成 `0` 或 `2`。

### 本程序的经验点

1. `Normalize` 前必须先 `ToTensor()`
2. `iter(train_loader)` 才能拿到 batch，`iter(train_data)` 只能拿到单样本

## 2_read_customdataset.py

### 程序作用

这个脚本演示了两件事：

1. 读取 MNIST 的原始二进制文件
2. 将 MNIST 转换成普通图片文件和标签索引文件，便于后续自定义数据集读取

### 关键流程

1. 用 `datasets.MNIST` 确认数据集存在
2. 用 `mnist.read_image_file` / `mnist.read_label_file` 读取原始数据
3. 将图像保存为 `.jpg`
4. 生成 `train.txt` 和 `test.txt`

### 关键函数与参数说明

#### `datasets.MNIST(root, train=True, download=False)`

参数解释：

- `root='./dataset/MNIST'`
	MNIST 根目录。
- `train=True / False`
	区分训练集与测试集。
- `download=False`
	假设数据已经存在；如果不存在需要改为 `True`。

#### `mnist.read_image_file(...)` 与 `mnist.read_label_file(...)`

这两个函数直接读取 `idx` 格式文件：

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

适用于理解 MNIST 原始存储格式，而不只是调用高级接口。

#### `convert_to_image(save_path, train=True)`

参数说明：

- `save_path`
	输出目录，比如 `./dataset/MNIST/mnist_data`
- `train=True`
	控制导出训练集还是测试集

函数内部设计说明：

- `os.makedirs(save_path, exist_ok=True)`
	先创建根目录，避免写 `train.txt` / `test.txt` 时出现路径不存在。
- `Image.fromarray(img.numpy(), mode='L')`
	`mode='L'` 表示单通道灰度图，适合 MNIST。
- `file_obj.write(f'{i}.jpg {int(label)}\n')`
	写入“文件名 标签”格式，方便后续用文本读取标签。

### 本程序的经验点

1. 自定义数据集通常要先把“图像文件”和“标签索引”组织好
2. 写标签文件时，路径和标签分隔格式必须与后续读取逻辑一致

## 3_build_selfdataset.py

### 程序作用

这个脚本演示如何基于本地图片和标签文件，自定义一个 `Dataset`，再交给 `DataLoader` 批量读取。

### 关键流程

1. 从 `train.txt` / `test.txt` 中读取图片路径和标签
2. 定义 `MnistDataset(Dataset)`
3. 在 `__getitem__` 中读取图像并返回 `(image, label)`
4. 用 `DataLoader` 批量取数据

### 关键类与函数说明

#### `class MnistDataset(Dataset)`

这是 PyTorch 自定义数据集最常见的写法，必须实现：

1. `__len__()`
2. `__getitem__(index)`

#### `__init__(self, image_path, image_label, transform=None)`

参数说明：

- `image_path`
	图片路径列表
- `image_label`
	标签列表
- `transform=None`
	图像增强方法

为什么这样设计：

- 将路径和标签分离保存，便于从不同来源构造数据集
- `transform` 作为可选参数，便于训练集和测试集使用不同增强策略

#### `__getitem__(self, index)`

处理步骤：

1. `Image.open(...)` 读取图像
2. `np.array(image)` 转成数组
3. 读取对应标签
4. 如果存在 `transform`，则对图像做处理
5. 返回图像和张量化标签

这里标签用了：

```python
torch.tensor(label)
```

说明：

- 这里 `label` 目前是浮点数，所以输出标签是浮点张量
- 如果后续是分类损失函数，例如 `CrossEntropyLoss`，更常见的是整数类别标签，通常会改成 `long` 类型

#### `get_path_label(img_root, label_file_path)`

示例：

```python
pd.read_csv(label_file_path, sep=r'\s+', names=['img', 'label'])
```

参数说明：

- `sep=r'\s+'`
	用任意空白符作为分隔符。
	因为标签文件是 `0.jpg 5` 这种空格格式，而不是逗号格式。
- `names=['img', 'label']`
	显式指定列名，避免没有表头时解析错误。

#### `DataLoader(dataset=train_dataset, batch_size=3, shuffle=True, num_workers=4)`

参数选值解释：

- `batch_size=3`
	这里只是教学演示，所以故意设小，便于打印一个 batch 时观察形状。
- `shuffle=True`
	训练集打乱。
- `shuffle=False`
	测试集保持固定顺序。
- `num_workers=4`
	表示用 4 个进程加载数据。
	在 Linux 下一般可行；Windows 常建议设为 `0`。

### 本程序的经验点

1. `Dataset` 负责“单样本读取逻辑”
2. `DataLoader` 负责“批量化、打乱、多进程加载”
3. 标签文件的分隔符必须与读取代码一致

## 4_ImageFolder.py

### 程序作用

这个脚本演示如何把二分类图片目录拆分成 `train/test`，然后直接用 `ImageFolder` 构建数据集。

### 关键流程

1. 从 `PetImages/Cat` 和 `PetImages/Dog` 读取图片
2. 按比例拆分到 `train/Cat`、`train/Dog`、`test/Cat`、`test/Dog`
3. 调用 `ImageFolder` 自动生成类别标签
4. 用 `DataLoader` 构建批量读取器

### 关键函数与参数说明

#### `split_dataset(train_ratio=0.8, seed=42)`

参数说明：

- `train_ratio=0.8`
	80% 训练集，20% 测试集。
	这是图像分类里常见的基础拆分比例。
- `seed=42`
	固定随机种子，保证每次拆分结果一致，便于复现实验。

函数内部关键点：

- `valid_suffixes = {'.jpg', '.jpeg', '.png', '.bmp'}`
	用于过滤非图片文件，例如 `Thumbs.db`。
- `random.shuffle(image_files)`
	先打乱再拆分，避免文件名排序带来分布偏差。
- `shutil.copy2(...)`
	复制文件到目标目录，并尽量保留元信息。

#### `ImageFolder(dataset_root, transform=...)`

使用要求：

```text
train/
	Cat/
	Dog/
test/
	Cat/
	Dog/
```

`ImageFolder` 会：

1. 自动遍历子目录
2. 把子目录名当作类别名
3. 自动建立 `class_to_idx`

这就是为什么这里不再需要单独的 `txt` 标签文件。

#### `build_dataloader(dataset_root, transform, shuffle)`

这里把 `ImageFolder` 和 `DataLoader` 包成了一个函数，优点是：

- 训练集和测试集构造方式统一
- 只需要通过 `shuffle` 控制是否打乱

#### 变换参数解释

```python
transforms.Normalize([0.485, 0.456, 0.406],
										 [0.229, 0.224, 0.225])
```

- 三通道彩色图像通常按 `RGB` 三个通道标准化
- 这里仍然采用 ImageNet 常见均值方差
- 这是分类模型训练中非常常见的默认预处理方式

### 本程序的经验点

1. `ImageFolder` 适合目录清晰的分类任务
2. 类别名来自子目录名，不来自文件名
3. 数据拆分前先打乱，能减少数据分布偏差

## 5_transform.py

### 程序作用

这个脚本用单张图片 `cat.jpg` 演示常见图像增强与预处理操作，并用子图对比显示结果。

### 已使用的变换与参数解释

#### `CenterCrop([200, 200])`

- 从图像中心裁出 `200 x 200`
- 适合演示“保留主体中心区域”的效果

#### `RandomCrop([200, 200])`

- 随机位置裁剪 `200 x 200`
- 常用于增加样本位置变化

#### `RandomResizedCrop(200, scale=(0.08, 1.0), ratio=(0.75, 1.55))`

参数说明：

- `200`
	输出尺寸为 `200 x 200`
- `scale=(0.08, 1.0)`
	随机裁剪区域面积占原图面积的比例范围
- `ratio=(0.75, 1.55)`
	裁剪区域宽高比范围

为什么这样选：

- `scale` 范围大，能模拟从局部到全图的不同观察尺度
- `ratio` 范围适中，避免过于极端的长条形裁剪

#### `RandomHorizontalFlip(0.7)`

- `0.7` 表示 70% 概率水平翻转
- 水平翻转通常比垂直翻转更常用，因为很多自然图像上下方向不能随便颠倒

#### `RandomVerticalFlip(0.8)`

- `0.8` 表示 80% 概率垂直翻转
- 教学演示里可以这样写，但真实任务里垂直翻转往往要谨慎使用

#### `RandomRotation(30)`

- 在 `[-30, 30]` 度之间随机旋转
- 30 度属于中等强度旋转增强

#### `Pad(10, fill=0, padding_mode='constant')`

参数说明：

- `10`
	四周补 10 个像素
- `fill=0`
	用黑色填充
- `padding_mode='constant'`
	表示常数填充，而不是反射填充或边缘复制

#### `ColorJitter(brightness=1, contrast=0.5, saturation=0.5, hue=0.4)`

参数说明：

- `brightness=1`
	亮度变化范围较大，演示效果明显
- `contrast=0.5`
	中等对比度扰动
- `saturation=0.5`
	中等饱和度扰动
- `hue=0.4`
	色相变化较明显，教学展示效果强，但实际训练中通常会取更小值

#### `Grayscale(1)`

- `1` 表示输出单通道灰度图

#### `RandomAffine(degrees=45, translate=(0.5, 0.7), scale=(0.5, 0.8), shear=3)`

参数说明：

- `degrees=45`
	随机旋转角度范围
- `translate=(0.5, 0.7)`
	水平最大平移 50%，垂直最大平移 70%
- `scale=(0.5, 0.8)`
	缩放比例从 0.5 到 0.8 之间随机采样
- `shear=3`
	剪切角度范围较小，用于轻量形变

注意：

- `scale` 必须满足前小后大，例如 `(0.5, 0.8)`
- 如果写成 `(0.8, 0.5)` 会直接报错

#### `Resize([100, 200])`

- 输出高度为 `100`，宽度为 `200`
- 用于演示非正方形缩放结果

#### `Normalize(mean, std)`

示例：

```python
mean = [0.45, 0.5, 0.5]
std = [0.3, 0.6, 0.5]
```

参数意义：

- `mean`
	每个通道减去的均值
- `std`
	每个通道除以的标准差

这里的值主要是为了演示标准化过程，不一定来自真实数据集统计值。

### 子图显示部分

`show_image_with_axes(...)` 这个辅助函数主要做了三件事：

1. 用 `plt.subplot(...)` 指定子图位置
2. 用 `plt.imshow(...)` 显示图像
3. 用 `plt.xticks(...)` 和 `plt.yticks(...)` 显示像素坐标

当前刻度间隔为 50 像素，因此更适合观察图像尺寸和变换结果。

## 6_summarize.py

### 程序作用

这个脚本相当于前面内容的综合版：

1. 定义一组 `Compose` 变换
2. 读取 CIFAR10 训练集和测试集
3. 构建训练与测试 `DataLoader`

### 设计意图

它把“数据集读取 + 数据增强 + 批量加载”串联起来，接近真实训练前的数据准备流程。

### 关键参数解释

#### `Resize((32, 32))`

- CIFAR10 原图本身就是 `32 x 32`
- 这里再次写 `Resize((32, 32))` 更多是为了展示完整预处理流程

#### `ColorJitter(0.3, 0.3, 0.2)`

- 比 `1_read_dataset.py` 稍强一些
- 说明综合脚本里增强强度可以适当增加

#### `RandomRotation(10)`

- 比 5 度更强，适合综合演示

#### `RandomAffine(...)`

这里脚本里当前参数仍然需要注意两个问题：

1. `scale` 如果写成前大后小会报错
2. `test_loader` 应该加载 `test_data`，而不是 `train_data`

也就是说，这个脚本更适合作为“章节综合练习模板”，在实际训练前仍应再检查一遍参数。

## 本章整体知识脉络

### 1. 标准数据集读取

对应：`1_read_dataset.py`

核心点：

- `torchvision.datasets` 直接读取
- `transform` 负责图像预处理
- `DataLoader` 负责 batch 化

### 2. 原始数据转本地图片

对应：`2_read_customdataset.py`

核心点：

- 把特殊格式数据转成普通图片
- 生成标签文件，便于后续自定义读取

### 3. 自定义 Dataset

对应：`3_build_selfdataset.py`

核心点：

- 自己控制单样本读取逻辑
- 适合标签来源灵活、目录结构不标准的数据集

### 4. ImageFolder 分类数据组织

对应：`4_ImageFolder.py`

核心点：

- 目录名就是类别名
- 非常适合标准图像分类任务

### 5. 图像增强与可视化

对应：`5_transform.py`

核心点：

- 学会区分几何变换与颜色变换
- 理解 `Normalize`、`ToTensor` 的顺序
- 用可视化验证增强是否合理

## 本章常见易错点

1. `Normalize` 前忘记加 `ToTensor()`
2. `train=True` / `train=False` 写错
3. `Dataset` 和 `DataLoader` 的作用混淆
4. 标签文件分隔符与读取代码不一致
5. `RandomAffine` 的 `scale` 范围写反
6. `ImageFolder` 目录结构不符合要求
7. 图像显示时忘记 `plt.show()`

## 参数选值的一般建议

### 批量大小

- `batch_size=32` 或 `64` 是最常见起点
- 显存不足时减小
- 教学打印时可以设成 `2` 或 `3`

### 线程数

- Linux 可先尝试 `num_workers=4`
- CPU 核较多时可增大
- Windows 建议从 `0` 开始测试

### 增强强度

- 小数据集先用轻量增强
- 颜色变换不要过强，否则会破坏语义
- 旋转、仿射、裁剪最好逐步加大强度，不要一开始就设过大范围

### 标准化参数

- 最理想的是使用当前数据集真实统计值
- 如果是迁移学习，常先用 ImageNet 默认均值方差

## 一句话总结

这一章的重点不是某一个 API，而是建立完整的数据处理链路：

从“读数据”开始，到“整理成图片和标签”，再到“自定义数据集 / ImageFolder”，最后用 `transforms` 和 `DataLoader` 把数据组织成可直接送入模型训练的形式。