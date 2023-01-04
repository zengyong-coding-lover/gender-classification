'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 19:39:12
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-07 19:59:32
FilePath: \gender-classification\models\XNet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn

def gaussian_noise(img, mean = 0.0, sigma = 0.1):
    noise = torch.normal(mean= mean, std= sigma, size = img.shape)
    gau = img + noise.to('cuda')
    return gau

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
class XNet_Noise(nn.Module):
    def __init__(self):
        super(XNet_Noise, self).__init__()
        b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # num_channels为当前的通道数
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # 输出为1维
        nn.Linear(num_channels, 1))
    def transition_block(self, input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    def forward(self, X):
        return self.net(gaussian_noise(X))
    def predict(self, X):
        return self.net( X)