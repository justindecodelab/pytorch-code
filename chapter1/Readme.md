# Chapter 1 Program Notes

本文档在原有说明基础上，补充 chapter_1 中每个程序涉及的主要函数、作用和关键参数含义，方便按脚本学习 PyTorch 基础知识。

## 1_tensor.py

这个脚本主要演示 Tensor 的创建、数据类型、形状读取和基础属性查看。

### 相关函数说明

`torch.empty(5, 3)`
- 作用：创建一个形状为 `(5, 3)` 的 Tensor。
- 特点：只分配内存，不初始化具体值，所以打印结果通常是未初始化数据。
- 参数：
	- `5, 3`：表示 Tensor 的 shape，也可以理解为 5 行 3 列。

`torch.ones(2, 3)`
- 作用：创建一个所有元素都为 1 的 Tensor。
- 参数：
	- `2, 3`：表示输出 Tensor 的维度。

`torch.zeros(2, 3)`
- 作用：创建一个所有元素都为 0 的 Tensor。
- 参数：
	- `2, 3`：表示输出 Tensor 的维度。

`torch.zeros(2, 3, dtype=torch.long)`
- 作用：创建整型全 0 Tensor。
- 参数：
	- `dtype=torch.long`：指定数据类型为长整型，常用于类别标签。

`torch.rand(5, 3)`
- 作用：创建 `[0, 1)` 上均匀分布的随机数 Tensor。
- 参数：
	- `5, 3`：输出 shape。

`torch.randn(4, 4)`
- 作用：创建服从标准正态分布的随机 Tensor。
- 参数：
	- `4, 4`：输出 shape。

`torch.tensor([5.5, 3])`
- 作用：根据已有数据直接构造 Tensor。
- 参数：
	- `[5.5, 3]`：初始数据。

`torch.tensor([5.5, 3], dtype=torch.double)`
- 作用：创建双精度浮点 Tensor。
- 参数：
	- `dtype=torch.double`：指定数据类型为 `float64`。

`x.new_ones(5, 3)`
- 作用：基于已有 Tensor `x` 的设备和数据类型创建全 1 Tensor。
- 参数：
	- `5, 3`：新 Tensor 的 shape。

`torch.randn_like(x, dtype=torch.float)`
- 作用：创建一个与 `x` 形状相同的随机 Tensor。
- 参数：
	- `x`：作为参考 Tensor，只复制其 shape。
	- `dtype=torch.float`：将数据类型设为 `float32`。

`x.size()` 或 `x.shape`
- 作用：获取 Tensor 的形状。
- 返回值：通常是 `torch.Size([...])`。

`torch.arange(1, 10, 1)`
- 作用：生成等差序列 Tensor。
- 参数：
	- `1`：起始值，包含。
	- `10`：结束值，不包含。
	- `1`：步长。

### 本脚本重点参数

- `dtype`：控制 Tensor 的数据类型。
- `shape / size`：描述 Tensor 每一维的长度。
- `torch.long`：常用于整数索引和分类标签。
- `torch.float`、`torch.double`：分别对应单精度和双精度浮点数。

## 2_operations.py

这个脚本主要演示 Tensor 的四则运算、矩阵乘法、切片、变形、拼接和数值裁剪。

### 相关函数说明

`torch.ones(5, 3)`
- 作用：构造全 1 Tensor，通常用于作为基础测试数据。

`torch.rand(5, 3)`
- 作用：构造随机 Tensor，便于测试加法等运算。

`x + y`
- 作用：执行逐元素加法。
- 要求：`x` 与 `y` 形状相同，或满足广播规则。

`torch.add(x, y)`
- 作用：与 `x + y` 等价，也是逐元素加法。

`torch.add(x, y, out=z3)`
- 作用：将加法结果直接写入 `z3`。
- 参数：
	- `out=z3`：输出保存到指定 Tensor，避免重复创建新内存。

`y.add_(x)`
- 作用：原地加法，执行后 `y` 本身会被修改。
- 特点：带下划线 `_` 的函数通常表示原地操作。

`torch.sub(a, b)`
- 作用：逐元素减法。

`torch.mul(a, b)`
- 作用：逐元素乘法。

`torch.div(a, b)`
- 作用：逐元素除法。

`torch.mm(a, b)`
- 作用：二维矩阵乘法。
- 参数要求：
	- 若 `a` 的 shape 为 `(m, n)`，则 `b` 的 shape 必须为 `(n, p)`。

`torch.abs(a)`
- 作用：求绝对值。

`torch.pow(a, 2)`
- 作用：计算幂。
- 参数：
	- `2`：指数，表示每个元素平方。

`x[:, 0]`
- 作用：切片取出第 0 列。
- 含义：
	- `:` 表示所有行。
	- `0` 表示第 1 列。

`x.view(16)`
- 作用：改变 Tensor 形状为一维长度 16。
- 注意：元素总数必须保持不变。

`x.view(-1, 2)`
- 作用：将 Tensor 重塑为列数为 2 的二维 Tensor。
- 参数：
	- `-1`：让 PyTorch 自动推导该维度大小。

`torch.cat((x, x), dim=0)`
- 作用：在第 0 维拼接多个 Tensor。
- 参数：
	- `(x, x)`：参与拼接的 Tensor 序列。
	- `dim=0`：按行方向拼接。

`torch.cat((x, x, x), dim=1)`
- 作用：在第 1 维拼接多个 Tensor。
- 参数：
	- `dim=1`：按列方向拼接。

`x.item()`
- 作用：把只有一个元素的 Tensor 转成 Python 标量。
- 适用场景：打印、日志记录、与普通 Python 数值交互。

`torch.clamp(a, -0.5, 0.5)`
- 作用：把 Tensor 元素限制在指定区间内。
- 参数：
	- `-0.5`：最小值。
	- `0.5`：最大值。

### 本脚本重点参数

- `out`：把计算结果写入已有 Tensor。
- `dim`：指定在哪个维度上拼接或取最大值。
- `-1`：在 `view` 中表示自动推导维度。
- `_` 后缀：表示原地操作，会直接修改原 Tensor。

## 3_numpy.py

这个脚本主要演示 PyTorch Tensor 和 NumPy ndarray 之间的相互转换，以及共享内存的特点。

### 相关函数说明

`torch.ones(5)`
- 作用：创建长度为 5 的一维 Tensor。

`a.numpy()`
- 作用：把 CPU 上的 Tensor 转换为 NumPy 数组。
- 特点：通常与原 Tensor 共享内存，修改一个，另一个也会变化。

`a.add_(1)`
- 作用：原地给 Tensor 中每个元素加 1。
- 结果：因为共享内存，`b` 对应的 NumPy 数组内容也会同步变化。

`np.ones(5)`
- 作用：创建长度为 5 的 NumPy 数组。

`torch.from_numpy(a)`
- 作用：把 NumPy 数组转换为 Tensor。
- 特点：与原始 NumPy 数组共享内存。

`np.add(a, 1, out=a)`
- 作用：给数组逐元素加 1，并把结果写回 `a`。
- 参数：
	- `a`：输入数组。
	- `1`：加数。
	- `out=a`：输出仍写入原数组。

### 本脚本重点参数

- `numpy()`：Tensor 转 ndarray。
- `from_numpy()`：ndarray 转 Tensor。
- `out`：表示原地覆盖输出。
- 共享内存：修改 Tensor 或 ndarray，另一个对象通常也会同步变化。

## 4_autograd.py

这个脚本分两部分：第一部分展示深度学习训练的一般反向传播流程；第二部分通过小例子演示自动求导机制。

### 相关函数说明

`torchvision.models.resnet18(pretrained=False)`
- 作用：创建 ResNet-18 模型。
- 参数：
	- `pretrained=False`：不加载预训练权重，使用随机初始化参数。
- 补充：在较新版本 torchvision 中常写成 `weights=None`，含义相近。

`torch.rand(1, 3, 64, 54)`
- 作用：生成模拟输入图片数据。
- 参数含义：
	- `1`：batch size。
	- `3`：通道数，表示 RGB 图像。
	- `64, 54`：图像高和宽。

`torch.rand(1, 1000)`
- 作用：生成模拟标签或目标输出。
- `1000` 对应 ResNet-18 默认 ImageNet 分类输出维度。

`model(data)`
- 作用：执行前向传播，得到预测结果 `prediction`。

`(prediction - labels).sum()`
- 作用：构造一个简单损失值。
- 说明：实际训练中通常会用更标准的损失函数，比如交叉熵或均方误差。

`loss.backward()`
- 作用：执行反向传播，自动计算可训练参数相对于损失的梯度。

`torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)`
- 作用：定义随机梯度下降优化器。
- 参数：
	- `model.parameters()`：需要更新的模型参数。
	- `lr=1e-2`：学习率，等于 `0.01`。
	- `momentum=0.9`：动量项，用于加快收敛、减小震荡。

`optim.step()`
- 作用：根据已经计算好的梯度更新模型参数。

`torch.ones((2, 2))`
- 作用：创建普通 Tensor，默认不跟踪梯度。

`torch.ones((2, 2), requires_grad=True)`
- 作用：创建需要梯度的 Tensor。
- 参数：
	- `requires_grad=True`：告诉 autograd 后续要追踪该 Tensor 的运算图。

`x.grad_fn`
- 作用：查看该 Tensor 是否由某个运算产生。
- 说明：叶子节点通常 `grad_fn` 为 `None`。

`x.is_leaf`
- 作用：判断 Tensor 是否为叶子节点。
- 意义：通常只有叶子节点会直接保存 `.grad`。

`y.backward()`
- 作用：对标量 `y` 求导，计算它对 `a`、`b` 的梯度。

`a.grad`、`b.grad`
- 作用：查看反向传播后得到的梯度值。

### 本脚本重点参数

- `requires_grad`：是否需要自动求导。
- `lr`：学习率，控制每次参数更新步长。
- `momentum`：动量参数。
- `parameters()`：返回模型中可训练参数。

## 5_linear_regression.py

这个脚本用最基础的方式手写一个一元线性回归模型，完整演示数据准备、定义模型、定义损失、反向传播和参数更新。

### 相关函数说明

`np.array(..., dtype=np.float32)`
- 作用：构造训练数据。
- 参数：
	- `dtype=np.float32`：指定 32 位浮点数，便于后续转换为 PyTorch Tensor。

`torch.from_numpy(x_train)`
- 作用：把 NumPy 数组转换为 Tensor。

`torch.tensor([-1.], requires_grad=True)`
- 作用：定义待学习参数 `w`。
- 参数：
	- `[-1.]`：参数初值。
	- `requires_grad=True`：训练时需要对该参数求梯度。

`torch.tensor([0.], requires_grad=True)`
- 作用：定义偏置参数 `b`。

`def linear_model(x): return x * w + b`
- 作用：定义线性模型 $y = wx + b$。
- 参数：
	- `x`：输入特征。

`def get_loss(y_, y): return torch.mean((y_ - y_train) ** 2)`
- 作用：计算均方误差损失。
- 数学形式：$MSE = \frac{1}{n}\sum (y_ - y)^2$。
- 说明：脚本中函数参数写了 `y`，但函数体实际使用的是外部变量 `y_train`。

`torch.mean(...)`
- 作用：求平均值，用于得到标量损失。

`loss.backward()`
- 作用：对损失执行反向传播，计算 `w.grad` 和 `b.grad`。

`w.data = w.data - lr * w.grad.data`
- 作用：手动更新参数 `w`。
- 参数：
	- `lr`：学习率。
	- `w.grad.data`：当前梯度。

`b.data = b.data - lr * b.grad.data`
- 作用：手动更新偏置参数 `b`。

`w.grad.zero_()`、`b.grad.zero_()`
- 作用：把上一轮累积的梯度清零。
- 原因：PyTorch 默认会对梯度做累加，不清零会影响下一轮结果。

`plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')`
- 作用：绘制真实样本点。
- 参数：
	- `'bo'`：蓝色圆点。
	- `label='real'`：图例名称。

`plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimate')`
- 作用：绘制模型拟合直线或预测点。
- 参数：
	- `'ro'`：红色圆点。
	- `label='estimate'`：图例名称。

`plt.legend()`
- 作用：显示图例。

`plt.show()`
- 作用：显示图像窗口。

### 本脚本重点参数

- `requires_grad=True`：让参数支持反向传播。
- `lr = 1e-2`：学习率。
- `epoch` / `e`：训练轮数计数。
- `zero_()`：原地清零梯度。

## 6_FC_MNIST_Classification.py

这个脚本实现了一个多层全连接神经网络，用于 MNIST 手写数字分类。

### 原 README 内容补充说明

#### 1. About layer width

`in_c = 784`
- 由 MNIST 数据集决定。
- 每张图片大小为 `28 x 28`，拉平成一维后为 `28 * 28 = 784`。

`out_c = 10`
- 由分类任务决定。
- 对应数字 `0` 到 `9` 共 10 个类别。

#### 2. About layer numbers / structure

`784 -> 512 -> 256 -> 128 -> 10`
- 表示神经网络每层神经元数量逐层递减。
- 这种结构属于经验性设计，常用于逐步提取特征并压缩表示。

#### 3. Training set / Test set

Training set
- 用于学习模型参数。
- 完整流程通常为：前向传播得到预测值 -> 计算损失 -> 反向传播计算梯度 -> 更新参数。

Test set
- 用于评估模型泛化能力。
- 一般只做前向传播和指标统计，不更新参数。

### 相关类与函数说明

`class Net(nn.Module)`
- 作用：定义全连接分类网络。
- 继承 `nn.Module` 后，模型可以自动管理参数，并支持 `.train()`、`.eval()` 等接口。

`def __init__(self, in_c=784, out_c=10)`
- 作用：初始化网络结构。
- 参数：
	- `in_c=784`：输入特征维度。
	- `out_c=10`：输出类别数。

`nn.Linear(in_c, 512)`
- 作用：定义全连接层。
- 参数：
	- `in_c`：输入特征数。
	- `512`：输出特征数，也就是本层神经元数量。

`nn.ReLU(inplace=True)`
- 作用：ReLU 激活函数，增加非线性表达能力。
- 参数：
	- `inplace=True`：尽量直接在原内存上修改，减少额外内存开销。

`def forward(self, x)`
- 作用：定义前向传播逻辑。
- 输入 `x` 会依次经过 `fc1 -> act1 -> fc2 -> act2 -> fc3 -> act3 -> fc4`。

`net = Net()`
- 作用：实例化网络。
- 默认使用 `in_c=784`、`out_c=10`。

`mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)`
- 作用：加载训练集。
- 参数：
	- `'./data'`：数据保存路径。
	- `train=True`：表示训练集。
	- `transform=transforms.ToTensor()`：把图片转换为 Tensor，并归一化到 `[0, 1]`。
	- `download=True`：若本地没有数据则自动下载。

`mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)`
- 作用：加载测试集。
- 参数：
	- `train=False`：表示测试集。

`DataLoader(train_set, batch_size=64, shuffle=True)`
- 作用：按批次加载训练数据。
- 参数：
	- `batch_size=64`：每个 batch 的样本数量。
	- `shuffle=True`：每个 epoch 前打乱数据顺序。

`DataLoader(test_set, batch_size=128, shuffle=False)`
- 作用：按批次加载测试数据。
- 参数：
	- `batch_size=128`：测试时 batch 可以适当更大。
	- `shuffle=False`：测试集通常不需要打乱。

`nn.CrossEntropyLoss()`
- 作用：多分类任务常用损失函数。
- 输入要求：
	- 模型输出是未经过 softmax 的 logits。
	- 标签是类别索引，不是 one-hot 向量。

`optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)`
- 作用：定义优化器。
- 参数：
	- `net.parameters()`：模型的可训练参数。
	- `lr=1e-2`：学习率。
	- `weight_decay=5e-4`：L2 正则项系数，用于抑制过拟合。

`net.train()`
- 作用：切换到训练模式。
- 意义：如果网络中有 Dropout、BatchNorm 等层，训练模式与测试模式行为不同。

`img.reshape(img.size(0), -1)`
- 作用：把图像从二维像素矩阵展平成一维向量。
- 参数含义：
	- `img.size(0)`：当前 batch 的样本数。
	- `-1`：自动推导每个样本展平后的维度，这里通常是 784。

`Variable(img)`、`Variable(label)`
- 作用：旧版本 PyTorch 中用于封装 Tensor 以支持自动求导。
- 补充：新版本 PyTorch 中一般可直接使用 Tensor。

`out = net(img)`
- 作用：执行前向传播，得到每个类别的得分。

`loss = criterion(out, label)`
- 作用：计算当前 batch 的分类损失。

`optimizer.zero_grad()`
- 作用：清空上一轮梯度。

`loss.backward()`
- 作用：执行反向传播。

`optimizer.step()`
- 作用：根据梯度更新模型参数。

`out.max(1)`
- 作用：在类别维上取最大值，得到预测类别。
- 参数：
	- `1`：表示按第 1 维，也就是类别维度取最大值。
- 返回：
	- 最大值。
	- 最大值所在索引，通常索引就是预测类别。

`(pred == label).sum().item()`
- 作用：统计当前 batch 预测正确的样本数。

`loss.item()`
- 作用：把单元素 Tensor 损失转为 Python 数值，便于打印和记录。

`plt.subplot(1, 2, 1)` 和 `plt.subplot(1, 2, 2)`
- 作用：创建 1 行 2 列的子图，用于分别绘制 loss 和 accuracy 曲线。
- 参数：
	- `1, 2`：表示总布局为 1 行 2 列。
	- `1` 或 `2`：表示当前子图编号。

### 本脚本重点参数

- `in_c`：输入维度。
- `out_c`：输出类别数。
- `batch_size`：每次送入网络训练的样本数。
- `shuffle`：是否打乱数据。
- `lr`：学习率。
- `weight_decay`：L2 正则化强度。
- `nums_epoch`：总训练轮数。

## 学习建议

建议按照以下顺序阅读和运行：

1. `1_tensor.py`：先理解 Tensor 的创建和基本属性。
2. `2_operations.py`：熟悉常见张量运算。
3. `3_numpy.py`：理解 Tensor 和 NumPy 的关系。
4. `4_autograd.py`：掌握自动求导和反向传播。
5. `5_linear_regression.py`：理解最基本的训练流程。
6. `6_FC_MNIST_Classification.py`：综合理解完整分类任务。