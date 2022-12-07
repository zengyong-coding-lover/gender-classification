'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 12:03:38
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-07 19:23:46
FilePath: \gender-classification\models\ResNet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch import nn
from torch.nn import functional as F
# class ResNet(nn.Module):
#     def __init__(self):
#         super(self).__init__()

class ResNet(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=True):
        super().__init__()
        strides= 1
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,padding=1)
        if(use_1x1conv):
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3=None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if(self.conv3):
            X=self.conv3(X)
        print(X.shape,Y.shape)
        Y+=X
        return F.relu(Y)