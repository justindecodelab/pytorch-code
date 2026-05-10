
# 假设图像大小为32x32，输入通道数为3（RGB图像），输出类别数为10
import torch
import torch.nn as nn
import torch.nn.functional as F

#Lenet network architecture
class LeNet(nn.Module):
    def __int__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #input_channels, output_channels, kernel_size, stride=1 ...
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2)) # convolutional layer 1 with ReLU activation followed by max pooling
            x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2)) # convolutional layer 2 with ReLU activation followed by max pooling

            x = x.view(x.size()[0], -1) # flatten the output of conv2 to feed into fully connected layers
            x = F.relu(self.fc1(x)) # fully connected layer 1 with ReLU activation
            x = F.relu(self.fc2(x)) # fully connected layer 2 with ReLU activation
            x = self.fc3(x) # output layer, no activation function since we will use CrossEntropyLoss which applies softmax internally
            return x
        
