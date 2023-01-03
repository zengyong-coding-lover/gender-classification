'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 09:40:42
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2023-01-02 15:06:54
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
import seaborn as sns
models = [ "ResNet", "XNet"]
num_epochs = 2 
batchsize = 50
device = 'cpu'
lr = 0.1
train_dir = './bitmoji-faces-gender-recognition/train.csv'
train_image_dir = './bitmoji-faces-gender-recognition/BitmojiDataset/trainimages/'
test_dir = './bitmoji-faces-gender-recognition/sample_submission.csv'
test_image_dir = './bitmoji-faces-gender-recognition/BitmojiDataset/testimages'

if __name__ == '__main__':
    transform = lambda X: torch.tensor(X, device=device, dtype=torch.float32)
    dataset = ImageDataset(train_dir, 
            train_image_dir, transform, transform)
    train_len, valid_len = int(len(dataset) * 0.8), int(len(dataset) * 0.2)
    train, valid = Data.random_split(dataset, [train_len, valid_len])
    #print(valid.shape)
    train_iter = Data.DataLoader(train, batchsize, shuffle=True)
    valid_iter = Data.DataLoader(valid, batchsize, shuffle=True)
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
                loss = (CELoss(yhat, y))
                allloss += float(loss)
                sgd.zero_grad()
                loss.backward()
                sgd.step()
                #print('%f' % loss)
            print('epoch %d loss is : %f' % (epoch + 1, allloss / len(train)))
            ALLLOSS.append(allloss / len(train))
            #if epoch % 2 == 0:
            with torch.no_grad():
                valid_yhat = []
                valid_y = []
                for validx, validy in valid_iter:
                    valid_yhat.append(Model(validx).reshape(validy.shape))
                    valid_y.append(validy)
                valid_yhat = logsist(torch.cat(valid_yhat, dim=-1))
                valid_y = torch.cat(valid_y, dim=-1)
                acc = Accurate(valid_yhat, valid_y)
                print('epoch %d: accurate %f (valid)' % (epoch+1, acc))
            ALLACC.append(acc)
        plt.title(model)
        plt.plot([i for i in range(len(ALLACC))], ALLACC)
        plt.plot([i for i in range(len(ALLLOSS))], ALLLOSS)
        plt.legend(['train_acc', 'train_loss'])
        plt.xlabel('epoch')
        plt.show()
        TP, FP, FN, TN = getMatrix(valid_yhat > 0.5, valid_y)
        C = [[TP, FN], [FP, TN]]
        sns.set()
        f, ax = plt.subplots()
        sns.heatmap(C, annot=True, ax=ax)
        ax.set_title(model + 'Mixed Matrix')
        ax.set_xlabel('predict')
        ax.set_ylabel('true')
        plt.show()
        roc = np.array(ROC(valid_yhat, valid_y))
        plt.plot(roc[:, 0], roc[:, 1])
        plt.title(model + 'roc')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        auc = np.array(AUC(valid_yhat, valid_y))
        plt.plot(auc[:, 0], auc[:, 1])
        plt.title(model + 'AUC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()
        # 以下对真实测试集进行预测，可以提交到kaggle上
        Test = TestImageDataset(test_dir, test_image_dir)
        test_iter = Data.DataLoader(Test, batchsize, shuffle=False)
        test_img_names = []
        test_Y = []
        for img_name, test_X in test_iter:
            test_img_names.extend(img_name)
            test_Y.append(Model(test_X).unsqueeze(-1))
        test_Y = logsist(test_Y)
        test_label = []
        for i in test_Y:
            label = 1 if i > 0.5 else -1
            test_label.append(label)
        result = pd.DataFrame(columns=['image_id', 'is_male'], data = [[i, j] for i, j in zip(test_img_names, test_label)])
        result.to_csv('./' + model +'output/sample_submission.csv')

