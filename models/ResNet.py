'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 12:03:38
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-08 00:31:34
FilePath: \gender-classification\models\ResNet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch import nn
from torch.nn import functional as F
# class ResNet(nn.Module):
#     def __init__(self):
#         super(self).__init__()

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 5, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(5), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(5, 5, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(5, 10, 2))
        #b4 = nn.Sequential(*resnet_block(10, 10, 2))
        #b5 = nn.Sequential(*resnet_block(10, 10, 2))
        self.net = nn.Sequential(b1, b2, b3,# b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(10, 1))
    def forward(self, X):
        return self.net(X)