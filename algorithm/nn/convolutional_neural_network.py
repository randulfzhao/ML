import torch
import torch.nn as nn
import torch.nn.functional as F

"""
这个 CNN 类包含两个卷积层和两个全连接层。卷积层使用 PyTorch 的 nn.Conv2d 类定义，其中参数 in_channels 和 out_channels 分别表示输入通道数和输出通道数，kernel_size 表示卷积核的大小，stride 表示步长，padding 表示是否需要对输入数据进行 zero-padding。

全连接层使用 PyTorch 的 nn.Linear 类定义，其中参数 in_features 和 out_features 分别表示输入特征数和输出特征数。

在前向传播过程中，我们使用 ReLU 激活函数来处理卷积层的输出，并使用最大池化层来减小输出的维度。

最后，我们使用 PyTorch 的 view 函数将输出的张量展平，然后通过两个全连接层得到最终输出。

这是一个简单的 CNN 模型的示例实现，您可以根据自己的需求进行扩展和修改。

例如，您可以添加更多的卷积层和全连接层，或者使用不同的卷积核大小和步长。

您还可以使用不同的激活函数和损失函数，或者使用不同的优化器来优化模型的训练过程。
"""


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
