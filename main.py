'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 09:40:42
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-07 20:24:12
FilePath: \pythoncode\深度学习\gender-classification\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils import *
from dataloader import *
from importlib import import_module
from torch.utils import  data as Data
import pandas as pd
import torch
from torch import nn

if __name__ == '__main__':
    models = [ "XNet", "ResNet" ]
    init_args = [(), (3, 1)]
    num_epochs = 200 
    batchsize = 256
    device = 'cpu'
    lr = 0.1
    data = pd.read_csv('./bitmoji-faces-gender-recognition/train.csv')
    y = data['is_male'][:20]
    for index, i in enumerate(y):
        if i == -1:
            y[index] = 0
    y = torch.tensor(y, dtype=torch.float32, requires_grad=False, device=device)
    X = load_img('./bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/', data['image_id'])
    X = torch.tensor(X[:20], dtype=torch.float32, requires_grad=False, device=device)
    X = X.permute(0, 3, 1, 2)
    #X = torch.tensor(load_img('./bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/', data['image_id']), dtype=torch.float32)
    #test_img = torch.tensor(load_test_img('./bitmoji-faces-gender-recognition/BitmojiDataset/testimages/'))
    print(X.shape)
    dataset = Data.TensorDataset(X, y)
    train_len, test_len = int(len(dataset) * 0.8), int(len(dataset) * 0.2)
    train, test = Data.random_split(dataset, [train_len, test_len])
    #print(test.shape)
    train_iter = Data.DataLoader(train, batchsize, shuffle=True)
    test_iter = Data.DataLoader(test, batchsize, shuffle=True)
    CELoss = nn.BCEWithLogitsLoss()
    for model in models:
        myfile = import_module('models.' + model)
        class_model = getattr(myfile, model)
        Model = class_model()
        sgd = torch.optim.SGD(Model.parameters(), lr=0.01)# 定义优化器
        for epoch in range(0, num_epochs):
            allloss = 0
            for X, y in train_iter:
                yhat = Model(X).reshape(y.shape)
                yhat = logsist(yhat)
               # print(y, yhat)
                loss = CELoss(yhat, y)
               # print(loss)
                allloss += loss
                sgd.zero_grad()
                loss.backward()
                sgd.step()
            print('epoch %d loss is : %f' % (epoch + 1, allloss))
            if epoch % 2 == 0:
                with torch.no_grad():
                    for testx, testy in test_iter:
                        test_yhat = Model(testx)
                        test_yhat = logsist(test_yhat)
                        print('epoch %d: accurate %f (test)' % (epoch, Accurate(test_yhat, testy)))