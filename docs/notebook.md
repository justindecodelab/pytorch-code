pytorch professional non:
1. tensor: multi-array
2. shape: tensor size 
3. dtype: data type
4. rand: 用于生成服从区间[0,1)均匀分布的随机张量
5. randn: 用于生成服从均值为0、方差为1正太分布的随机张量(random normal)
5. randint​:随机生成int类型数据(random integer)
6. mul: 用来计算张量间的点乘
7. mm: 矩阵乘法计算
8. pow: power 乘方或幂
9. x[:, 1]: 使用Numpy-like的索引/切片操作
	:​ 表示所有（在第一个维度，即行维度上选择所有行)
	1​ 表示索引为1（在第二个维度，即列维度上选择第2列）
10. view: 类似resize/reshape操作，其中使用-1时pytorch自动根据其他维度进行推导
11. cat：concate操作根据指定的维度索引（dim）来拼接张量，拼接的数量由输入张量x的个数决定
12. item(): 比如将一个非pytorch类型数据变成float的数据类型
13. clamp: 限制取值范围(value, min, max)
14. resnet18:图像分类模型
15. autograd: 负责"记录公式"（如何求导步骤）
15. backward(): 负责"执行计算"（实际求导并得到数值）
16. SGD:Stochastic Gradient Descent随机梯度下降（优化器optim）
17. momentum: 引入惯性，使更新方向考虑历史梯度，加速收敛并减少振荡。推荐值0.9
18. lr(learning rate):控制参数更新的步长大小，决定每次梯度下降的幅度。决定训练其收敛问题。	推荐值0.001~0.1
18. weight_decay:权重衰减，# 规则：学习率大 → 衰减小；学习率小 → 衰减可稍大。调参:不确定时，从0.01​或0.001​开始试
19. .grad:梯度值(结果）
20. is_leaf:是否为叶子节点
21. loss: 预测结果prediction和标准结果label之间的差异

22. FC(full-connected layer):全连接层，是神经网络中的一种基本结构。表示神经网络中除输入层之外的每个节点都和上一层的所有节点有连接。
23. 全连接神经网络：也叫做多层感知机（Multilayer Perceptron，MLP）。输入层、隐藏层、输出层构成。
24. activate function:常用的Sigmoid、tanh、Relu，主要来进行非线性变换。
25. Net():在构造函数中新建一个神经网络
26. net():神经网络模型实例，执行前向传播。输入数据img的shape通常为:(batch_size, channels, height, width)
27. Variable():包装张量，使其支持自动求导。
28. enumerate:枚举，常见写法：for batch, (img, label) in enumerate(train_data): # batch 是索引，img 是数据，label 是标签
29. transforms:实现数据增强函数
30. transforms.Compose:compose组合函数是将所有增强的方法进行先后顺序排列。将数据增强方法存放在一个list中，最后传入Compose中。
31. Dataset:Dataset是对本地数据读取逻辑的定义
32. DataLoader:而DataLoader是对Dataset对象的封装，执行调度，将一个batch size的图像数据组装在一起，实现批量读取数据。
33. ImageFolder:分类任务通用的ImageFolder读取形式。如：训练集和测试集中分别包含有cat、dog、duck、horse四类图像的子文件夹。
34. 数据增广方法:一般会从图像颜色、尺寸、形态、亮度/对比度、噪声和像素等角度进行变换。不同的数据增广方法可以自由进行组合，得到更加丰富的数据增广方法。

transform方法如下：
1）裁剪效果：
35. CenterCrop:中心裁剪
36. RandomCrop:随机裁剪
37. RandomResizedCrop:随机长宽比裁剪

2）翻转和旋转效果：
38. RandomHorizontalFlip:依概率p水平翻转
39. RandomVerticalFlip:依概率p垂直翻转
40. RandomRotation:随机旋转

3）其他图像变换效果：
41. Pad:图像填充
42. ColorJitter:调整亮度、对比度和饱和度
43. Grayscale:转成灰度图
44. RandomAffine:仿射变换是线性变换（如旋转、缩放、剪切）加上平移变换的组合。
45. Resize:尺寸缩放
46. Compose:所有transform方法的组合

## 1.经典图像分类模型
46. CNN(Convolution Nerual Network):是解决图像分类、图像检索、物体检测和语义分割的主流模型。
### CNN基础
1）输入层：RGB输入，通道数(channel)也是深度(depth)。
2）二层卷积：receptive filed(感受野)和 filter(卷积核)进行相乘再相加的过程。本质上是一次加权求和，即用卷积核对局部图像的像素做加权平均(或特征提取）。
记忆口诀:卷积 = （感受野 × 卷积核）逐元素乘加 + 通道专属偏置(bias)
receptive filed 和 filter尺寸相等，其中前者表示图像像素、后者表示权重、bias表示专属通道的偏置。

多输入通道需要一组卷积核进行卷积操作，得到一个通道输出
多输出通道需要多组卷积核进行卷积操作，得到多个通道输出

注:中间结果(intermidiate层)=(感受野x卷积核)逐元乘加

3）填充(padding):激活图的边界处保存数据，从而提高性能。最常用的方法是零填充。
4）步长:(stride):步幅表示卷积核一次应移动多少像素。
5）激活函数(activate function)
Relu: 数学定义:ReLU(x)=max(0,x). 如果输入 x≥0，输出为 x；如果 x<0，输出为 0。
功能：通常作为卷积层或全连接层后的激活函数，引入非线性，使网络能够拟合非线性函数。

Softmax: 数学定义: $$ \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$
功能：确保CNN输出的总和为1。通常作为神经网络的最后一层，配合交叉熵损失函数训练分类模型。

6）池化层：目的都是要逐渐减小网络的空间范围，从而减少网络的参数和总体计算

池化（Pooling）​ 是CNN中的下采样操作
主要作用：
减小特征图尺寸，降低计算量
增大感受野，提取更抽象的特征
引入平移不变性，增强鲁棒性
防止过拟合

47. LeNet
### 网络架构
分为卷积层块和全连接层块两个部分：输入 → (卷积层 + 激活函数 + 池化层) × N → 展平 → 全连接层 → 输出
卷积层：用于提取空间特征。
池化层（下采样）：用于降低数据维度，增加平移不变性。
全连接层：在最后进行汇总和分类。
N：表示卷积层块的数量。

### function
Conv2d:2维卷积层，其主要参数是in_channels、out_channels、kernel_size。
max_pool2d:池化，主要输入激活函数Relu、conv卷积层、卷积核大小。
Linear:全连接层，Linear(16 * 5 * 5, 120)主要参数表示in_features(channel=16、height=width=5)，out_features=120，bias=True(默认包含偏置项）
view:platten(展平)，改变tensor形状(shape).eg:
卷积层输出: (batch, channels, height, width)
                    ↓
                展平操作
                    ↓
全连接层输入: (batch, channels × height × width)

具体例子:
输入: (4, 16, 5, 5)
    ↓ x.view(4, -1)
输出: (4, 400)  # 400 = 16 × 5 × 5

Dropout:随机丢弃，默认p=0.5。# 每个神经元有50%概率被保留

48. AlexNet
### 网络架构
包含特征提取器（features）和分类器（classifier）两部分。
特征提取器：
多层级结构排序:卷积层 + 激活函数 + 池化 或 无池化

分类器：
多层级结构排序:Dropout + full_connection + ReLU激活 —> num_classes(输出分类结果)

49. VGG
### 网络架构
input——>{[conv1_1]——>[conv1_1]——>[max-pool]} ——> {[conv2_2]——>[conv2_2]——>[max-pool]}——>[fc 4090]——>[fc 4090]——>[fc 1000]

VGG16:表示13个卷积层 + 3个全连接层
卷积核尺寸：(3 x 3)
VGG的深度 = 卷积层数量 + 全连接层数量

50. NiN
### 网络架构
前三层是MLP卷积层，最后一层是全局平均池化。
即串联多个由卷积层和“全连接”层构成的⼩网络，又称MLP卷积，来构建⼀个深层网络。

51. GoogLeNet
### 网络架构
Inception module: Previous layer——>{[1x1 convolutions] [3x3 convolution] [5x5 convolution] [3x3 max pool]}——>Filter concatenation

Sequential: 多层网络结构的排序函数


52. 批量归一化（Batch Normalization）
作用：模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

1）对全连接层做批量归一化：
在全连接层中，BN 通常置于仿射变换（Wu+b）与激活函数（ϕ）之间。
设定：
全连接层输入：u
权重和偏置参数：W,b
仿射变换输出：x=Wu+b
激活函数：ϕ
批量归一化运算符：BN
最终输出：ϕ(BN(x))

计算步骤:参考batch_normalization.png

关于可学习参数beta和gamma:beta表示拉伸(scale)、gamma表示偏移(shift).
这两个参数允许模型“撤销”归一化。
如果模型发现原始分布更好，它可以学习出 $\boldsymbol{\gamma} = \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}$ 和 $\boldsymbol{\beta} = \boldsymbol{\mu}_\mathcal{B}$，从而还原回原始输入


2）卷积层（Convolutional Layer）
位置： 卷积计算之后，激活函数之前。
作用范围： 对**每个通道（Channel）**分别做 BN。
计算细节： 假设通道输出大小为 $p \times q$，小批量大小为 $m$。我们需要对该通道内所有的 $m \times p \times q$ 个元素共同计算均值和方差。该通道共享一组标量参数 $\gamma$ 和 $\beta$。

3）预测模式（Inference/Prediction）
在预测阶段，我们通常只有一个样本，无法计算“小批量”的均值和方差。
策略： 使用移动平均（Moving Average）。
方法： 在训练过程中，记录整个数据集均值和方差的估算值。
结果： 确保模型在预测时输出是确定的，不依赖于预测时的批量大小。

批量归一化通过将每一层的中间输出限制在稳定的分布范围内，解决了深层网络难以训练的问题。与**残差网络（ResNet）**共同构成了现代深度学习架构的两大基石。

53. 残差网络(ResNet)
目的:解决的是深度神经网络的“退化”问题。“退化”指的是，给网络叠加更多的层后，性能却快速下降的情况。
残差 = 真实目标映射 f(x)减去输入 x，表达式：
		residual = f(x) - x
x:相当于一个目标值
f(x):真实值
residual:根据实时的f(x)与目标值x计算得到每次的value，通过微调来实现修正。


54. 自定义模型训练与验证
1. 数据预处理、加载
AI数据主要包括：文本、图像、音频、视频数据：
图像数据，常用OpenCv，Pillow包
音频数据，常用scipy，librosa包
文本数据，常用NLTK，SpaCy包

预处理阶段：包括数据增强等。

2. 模型训练、调参
2.1 定义神经网络架构
### 知识点
2.1.1 卷积层
self.conv1 = nn.Conv2d(3,8,3,padding=1)
定义一次卷积运算，其中第一个3表示输入为3通道对应到本次测试为图片的RGB三个通道，数字8的意思为8个卷积核，第二个3表示卷积核的大小为3x3，padding=1表示在图片的周围增加一层像素值用来保存图片的边缘信息。

official format:
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
in_channels(int) ：输入信号的通道
out_channels(int) ：卷积产生的通道
kerner_size(int or tuple) ：卷积核的尺寸
stride(int or tuple, optional) ：卷积步长
padding(int or tuple, optional) ：输入的每一条边补充0的层数
dilation(int or tuple, optional) ：卷积核元素之间的间距
groups(int, optional) ：从输入通道到输出通道的阻塞连接数
bias(bool, optional) ：如果bias=True，添加偏置

2.1.2 池化层
self.pool1 = nn.MaxPool2d(2,2)
二维池化其中第一个2表示池化窗口的大小为2x2，第二个2表示窗口移动的步长。

official format:
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
kernel_size(int or tuple) ：max pooling的窗口大小
stride(int or tuple, optional) ：max pooling的窗口移动的步长。默认值是kernel_size
padding(int or tuple, optional) ：输入的每一条边补充0的层数
dilation(int or tuple, optional) ：一个控制窗口中元素步幅的参数
return_indices ： 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作

2.1.3 归一化
self.bn1 = nn.BatchNorm2d(64)

official format:
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
num_features： 来自期望输入的特征数，该期望输入的大小为batch_size x num_features [x width]
eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
momentum： 动态均值和动态方差所使用的动量。默认为0.1。
affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数。
track_running_stats：布尔值，当设为true，记录训练过程中的均值和方差；

个人见解：
epsilon: eps在数学书通常表示极小的正数。

2.1.4 ReLU激活函数
self.relu1 = nn.ReLU()

Relu激活函数（The Rectified Linear Unit）修正线性单元，用于隐层神经元输出。

2.1.5 全连接层
self.fc14 = nn.Linear(512x4x4,1024)

全连层输入数据个数为一维数据512x4x4，输出个数为1024.

2.1.5 Dropout
self.drop1 = nn.Dropout2d()

默认p=0.5，删除掉隐藏层随机选取的一半神经元。防止过拟合overfitting


2.2 定义损失函数及优化
    loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate) # Adam 优化算法是随机梯度下降算法的扩展式

优化器:SGD优化器(随机梯度下降)

2.3 定义训练、验证、预测模块
# train_loader， 一次性加载了sample中全部的样本数据，每次以batch_size为一组循环
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sample, num_workers=2)   return train_loader

val_loader = torch.utils.data.DataLoader(train_set, batch_size=64, sampler=validation_sample, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sample, num_workers=2)

2.5 迭代训练并验证: reference code

3. 模型保存、加载
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load(' model.pt'))


## 目标检测
## 3.1 目标检测概念
### 3.1.1 定义
图像分类：只需要判断输入的图像中是否包含感兴趣物体。
目标检测：需要再识别出图片中目标类别的基础上，还需要精确定位到目标的具体位置，并用外接矩形框标出。
常见目标检测算法：YOLO、SSD、Faster R-CNN等。

### 3.1.2 思路
方案思路：先确立众多候选框，再对候选框进行分类和微调。
边界框：(class,x1,y1,x2,y2) 

### 3.1.3 目标框定义方式
目标框定义方式：
任何图像任务的训练数据都包含两项：图片和真实标签信息，通常叫做GT(Ground truth)。

外接边框：bounding box，bbox主要格式，(x1,y1,x2,y2)和(c_x,c_y,w,h)
图像坐标原点(0,0)通常在左上角。
Label:(C,x1,y1,x2,y2) 
其中，C表示框内物体类别，(x1,y1)表示左上角坐标、(x2,y2)表示右下角坐标。

Label:(C,x_c,y_c,w,h)
其中，C表示框内物体类别，(x_c,y_c)表示矩形框中心点坐标，(w,h)表示矩形框宽高。

注：两种标注数据格式可相互转换

### 3.1.4 交并比（IoU）
IoU的全称是交并比（intersection over Union），表示两个目标框的交集占其并集的比例。

计算流程简述：
1.首先获取两个框的坐标，红框坐标: 左上(red_x1, red_y1), 右下(red_x2, red_y2)，绿框坐标: 左上(green_x1, green_y1)，右下(green_x2, green_y2)
2.计算两个框左上点的坐标最大值:(max(red_x1, green_x1), max(red_y1, green_y1)), 和右下点坐标最小值:(min(red_x2, green_x2), min(red_y2, green_y2))
3.利用2算出的信息计算黄框面积：yellow_area
4.计算红绿框的面积：red_area 和 green_area
5.iou = yellow_area / (red_area + green_area - yellow_area)

## 3.2 目标检测数据集VOC
### 3.2.1 VOC数据集简介
数据集说明：
1）JPEGImages:存放所有的图片，包括训练验证测试用到的所有图片。
2）ImageSets:包含三个子文件夹，Layout、Main、Segmentation
Layout文件夹中存放的是train，valid，test和train+valid数据集的文件名
Segmentation文件夹中存放的是分割所用train，valid，test和train+valid数据集的文件名
Main文件夹中存放的是各个类别所在图片的文件名，比如cow_val，表示valid数据集中，包含有cow类别目标的图片名称。

3）Annotations:存放着每张图片相关的标注信息，以xml格式的文件存储。
filename：图片名称
size：图片宽高，
depth表示图片通道数
object：表示目标，包含下面两部分内容。
首先是目标类别name为dog。pose表示目标姿势为left，truncated表示是否是一个被截断的目标，1表示是，0表示不是，在这个例子中，只露出狗头部分，所以truncated为1。difficult为0表示此目标不是一个难以识别的目标。
然后就是目标的bbox信息，可以看到，这里是以[xmin,ymin,xmax,ymax]格式进行标注的，分别表示dog目标的左上角和右下角坐标。
一张图片中有多少需要识别的目标，其xml文件中就有多少个object。上面的例子中有两个object，分别对应人和狗。

### 3.2.2  VOC数据集的dataloader的构建
参考代码


## 3.3 锚框 or 先验框
### 3.3.1 关于先验框
1）设置不同尺度的先验框
一张图片中通过设置不同的尺度的先验框，就有更高的概率出现对于目标物体有良好匹配度的先验框。

2）先验框与特征图的对应
将先验框的设置位置与特征图建立一一对应的关系，减少先验框数量。通过特征图，直接一次性的输出所有先验框的类别信息以及坐标信息，提升预测速度。

3）先验框类别信息的确定
这些先验框中有很多是和图片中要检测的目标完全没有交集或者有很小的交集。做法：设定一个IoU阈值，例如iou=0.5，与图片中目标的iou<0.5的先验框，这些框我们将其划分为背景，Iou>=0.5的被归到目标先验框，通过这样划分，得到供模型学习的ground truth信息。

### 3.3.2 先验框的生成
生成过程：
0. cx， cy表示中心点坐标
1. 遍历特征图上每一个cell，i+0.5是为了从坐标点移动至cell中心，/fmap_dims目的是将坐标在特征图上归一化
2. 这个时候我们已经可以在每个cell上各生成一个框了，但是这个不是我们需要的，我们称之为base_prior_bbox基准框。
3. 根据我们在每个cell上得到的长宽比1:1的基准框，结合我们设置的3种尺度obj_scales和3种长宽比aspect_ratios就得到了每个cell的9个先验框。
4. 最终结果保存在prior_boxes中并返回。

需要注意的是，这个时候我们的到的先验框是针对特征图的尺寸并归一化的，因此要映射到原图计算IOU或者展示，需要：
img_prior_boxes = prior_boxes * 图像尺寸


## 3.4 模型结构
### 3.4.1 VGG16作为backbone
使用VGG16作为主干网络来提取特征图(fearture map)

其中anchor配置如下：
将原图均匀分成7x7个cell，每个cell表示一个小方格
设置3种不同的尺度：0.2, 0.4, 0.6
设置3种不同的长宽比：1:1, 1:2, 2:1

注：一张图是7x7的feature_map，因为设置3个obj_scales和3个aspect_ratios，所以一个cell包含3x3个anchor=9，总的anchor数 = 7x7x9 =  441。

### 3.4.2 分类头和回归头
3.4.2.1 边界框的编解码
定义：模型要预测anchor与目标框的偏移，并且这个偏移会进行某种形式的归一化。
Bounding Box with coordinates:(cx,cy,w,h) 
Prior with center-size coordinates(^cx,^cy,^w,^h)

ground_truth delta(真实偏移量): (g_cx,g_cy,g_w,g_h)
具体公示表达如下：
​ $$g_{cx}=\frac{c_x-\hat{c}_x}{\hat{w}}$$
​ $$g_{cy}=\frac{c_y-\hat{c}_y}{\hat{h}}$$
​ $$g_w=log(\frac{w}{\hat{w}})$$
​ $$g_h=log(\frac{h}{\hat{h}})$$

注：编码(encode)是模型预测并输出得到的ground_truth delta，按照公式反向进行解码(decode)得到bounding box

3.4.2.2 分类头与回归头预测
对于输出7x7的feature map上的每个先验框我们想预测：
1）边界框的一组21类分数，其中包括VOC的20类和一个背景类。
2）边界框编码后的偏移量($g_{cx},g_{cy},g_w,g_h$)。
为了得到我们想预测的类别和偏移量，我们需要在feature map后分别接上两个卷积层：
1）一个分类预测的卷积层采用3x3卷积核padding和stride都为1，每个anchor需要分配21个卷积核，每个位置有9个anchor，因此需要21x9个卷积核。
2）一个定位预测卷积层，每个位置使用3x3卷积核padding和stride都为1，每个anchor需要分配4个卷积核，因此需要4x9个卷积核。

注：回归头表示预测定位(location)，分类头表示预测分类。
卷积层是由很多个卷积核 x 感受野并逐相加组成的。卷积核表示weight，感受野表示像素点。
卷积核数量 = 输出通道数量
分类层预测卷积：21个卷积核表示20类+1背景类，一个anchor需要预测21类，一个位置=9x21=189个卷积核。
定位层预测卷积：4个卷积核表示对应4个通道，这4个通道在某一个anchor上分别对应[dx,dy，dw，dh]中的一个。一个anchor需要4类，预测有9组，channels=36。

每个卷积核就像一个“专门的测量员”：
卷积核 A：专门负责在全图扫描，预测每个位置的 $\Delta x$。
卷积核 B：专门负责扫描并预测 $\Delta y$。
卷积核 C 和 卷积核 D：分别负责预测 $\Delta w$ 和 $\Delta h$。

个人理解：
3x3卷积核表示大小为3x3像素的filter/kernel；
像素：表示在一张图片上的坐标(x,y)区域;
像素点：表示在(x,y)上的值；
分辨率：一张图片的大小尺寸；
神经元：全连接层(Fully Connected Layer)是由一个个神经元构成。


模型输出的shape应该为：
分类头 batch_size x 7 x 7 x 189
回归头 batch_size x 7 x 7 x 36

为了方便后面的处理将每个anchor的预测独自成一维，也就是：
分类头 batch_size x 441 x 21
回归头 batch_size x 441 x 4

441 = 7x7x9 先验框。


## 3.5 损失函数
### 3.5.1 Matching strategy (匹配策略)：
第一个原则：以ground truth box为基准，遍历prior bbox进行匹配。如果groundtruth box与prior bbox对应起来(有最大的jaccard overlap就是IOU)。如果没有一个prior bbox能与所有的ground truth进行匹配，那么prior bbox只能与背景进行匹配，就是负样本。

第二个原则：从prior bbox出发，对剩余的还没有配对的prior bbox与任意一个ground truth box尝试配对，只要两者之间的jaccard overlap大于阈值（一般是0.5），那么该prior bbox也与这个ground truth进行匹配。ground truth可能与多个Prior box匹配，反之则不行，且prior bbox只能与IoU最大的ground truth进行匹配。

注：第二原则一定在第一原则之后进行。如果某个ground truth所对应最大IoU的prior bbox小于阈值，并且所匹配的prior bbox却与另外一个ground truth的IoU大于阈值。该prior bbox必须与前者进行匹配，因为每一个ground truth一定要与最大的prior bbox进行匹配。


### 3.5.2 损失函数
总体的目标损失函数定义为 定位损失（loc）和置信度损失（conf）的加权平衡：
$$ L(x,c,l,g) = \frac{1}{N}(L_{conf}(x,c)+\alpha L_{loc} (x,l,g)) (1) $$

归一化因子N：这是这是匹配到真实框（Ground Truth, GT）的先验框（Prior Boxes）数量。将总损失除以 N，使得损失的大小与每张图片中目标的数量无关，保证了训练的稳定性。如果 N=0，损失直接设为 0。
权重系数 $\alpha$：这是一个超参数，用于平衡分类任务和回归任务。
如果 $\alpha$ 过大，模型会更关注定位精度，但可能导致分类错误。
如果 $\alpha$ 过小，模型分类很准，但预测框可能对不齐目标。实验证明 $\alpha=1$ 在大多数场景下效果最好。

### 3.5.3 Hard negative mining(难负例挖掘策略）
核心思想：在众多的背景区域（负样本）中，只挑选出那些“最容易让模型出错”的样本来进行训练，而忽略掉那些一眼就能看出是背景的“简单样本”。

## 3.6、训练与测试
### 3.6.1 模型训练
目标检测网络的训练大致是如下的流程：
1. 设置各种超参数
2. 定义数据加载模块 dataloader
3. 定义网络 model
4. 定义损失函数 loss
5. 定义优化器 optimizer
6. 遍历训练数据，预测-计算loss-反向传播

### 3.6.2 后处理
### 3.6.2.1 目标框信息解码
目的：对模型的回归头的输出进行解码，得到真正意义上的目标框的预测结果。

### 3.6.2.2 NMS非极大值抑制(Non-Maximum Suppression)
剔除冗余的检测框，确保同一个目标只保留一个最准确的预测结果。

NMS的大致算法步骤如下：
1. 按照类别分组，依次遍历每个类别。
2. 当前类别按分类置信度排序，并且设置一个最低置信度阈值如0.05，低于这个阈值的目标框直接舍弃。
3. 当前概率最高的框作为候选框，其它所有与候选框的IOU高于一个阈值（自己设定，如0.5）的框认为需要被抑制，从剩余框数组中删除。
4. 然后在剩余的框里寻找概率第二大的框，其它所有与第二大的框的IOU高于设定阈值的框被抑制。
5. 依次类推重复这个过程，直至遍历完所有剩余框，所有没被抑制的框即为最终检测框。

### 3.6.2.3 代码实现:
参考：整个后处理过程的代码实现位于model.py中tiny_detector类的detect_objects函数中

### 3.6.3 单图预测推理
当模型已经训练完成后，导入必要的python包，然后加载训练好的模型权重。

### 3.6.4 VOC测试集评测
### 3.6.4.1 map指标(Mean Average Precision,平均精度均值）
一级指标：positive表示相关性，negative表示非相关性
1）真实值是positive，模型认为是positive的数量（True Positive=TP）
2）真实值是positive，模型认为是negative的数量（False Negative = FN）：这就是统计学上的第二类错误（Type II Error）
3）真实值是negative，模型认为是positive的数量（False Positive = FP）：这就是统计学上的第一类错误（Type I Error）
4）真实值是negative，模型认为是negative的数量（True Negative = TN）

二级指标：
1）准确率（Accuracy）-----针对整个模型：分类模型所有判断正确的结构占总观测值的比重
2）精确率（Precision）：在模型预测是Positive的所有结果中，模型预测对的比重
3）灵敏度（Sensitivity）=（Recall）:在真实值是Positive的所有结果种，模型预测对的比重
4）特异度（Specificity）：在真实值是Negative的所有结果中，模型预测对的比重











