'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 09:40:42
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-08 00:35:11
FilePath: \gender-classification\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from utils import *
from dataloader import *
from importlib import import_module
from torch.utils import  data as Data
import torch
from torch import nn
from matplotlib import pylab as plt

if __name__ == '__main__':
    models = ["ResNet", "XNet"]
    init_args = [(), (3, 1)]
    num_epochs = 3 
    batchsize = 50
    device = 'cpu'
    lr = 0.1
    transform = lambda X: torch.tensor(X, device=device, dtype=torch.float32)
    dataset = ImageDataset('./bitmoji-faces-gender-recognition/train.csv', 
            r'.\bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/', transform, transform)
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
        Model = Model.to(device)
        sgd = torch.optim.SGD(Model.parameters(), lr=0.01)# 定义优化器
        ALLLOSS = []
        ALLACC = []
        for epoch in range(0, num_epochs):
            allloss = 0
            for X, y in train_iter:
                yhat = Model(X).reshape(y.shape)
                yhat = logsist(yhat)
                loss = CELoss(yhat, y)
                allloss += loss
                sgd.zero_grad()
                loss.backward()
                sgd.step()
                print('%f' % loss)
            print('epoch %d loss is : %f' % (epoch + 1, allloss / len(train)))
            ALLLOSS.append(allloss)
            #if epoch % 2 == 0:
            with torch.no_grad():
                test_yhat = []
                test_y = []
                for testx, testy in test_iter:
                    test_yhat.append(Model(testx).reshape(testy.shape))
                    test_y.append(testy)
                test_yhat = logsist(torch.cat(test_yhat, dim=-1))
                test_y = torch.cat(test_y, dim=-1)
                acc = Accurate(test_yhat, test_y)
                print('epoch %d: accurate %f (test)' % (epoch+1, acc))
            ALLACC.append(acc)
        plt.plot([i for i in range(len(ALLACC))], acc)
        plt.plot([i for i in range(len(ALLLOSS))], acc)
        plt.legend(['train_acc', 'train_loss'])
        plt.xlabel('epoch')
        plt.show()
        TP, FP, FN, TN = getMatrix(test_yhat, test_y)
        plt.heatmap([[TP, FN], [FP, TN]])
        plt.title(model + 'Mixed Matrix')
        plt.show()
        roc = ROC(test_yhat, test_y)
        plt.plot(roc[:, 0], roc[:, 1])
        plt.title(model + 'roc')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        auc = AUC(test_yhat, test_y)
        plt.plot(auc[:, 0], auc[:, 1])
        plt.title(model + 'AUC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()


