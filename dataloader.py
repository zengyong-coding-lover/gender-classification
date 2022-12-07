'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 10:22:11
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-07 11:18:40
FilePath: \gender-classification\dataloader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from matplotlib import pylab as plt
import os
#import pandas as pd
def load_img(img_path, img_names):
    X = []
    for img_name in img_names:
        X.append(plt.imread(img_path + img_name) / 255)
    return X 
def load_test_img(img_path):
    files = os.listdir(img_path)
    test = []
    for file in files:
        test.append(plt.imread(img_path + file) / 255)
    return test        