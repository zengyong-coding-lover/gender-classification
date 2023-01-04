
import numpy as np
import torch
def getMatrix(y_predict, y_true):
    TP = FP = TN = FN = 0
    for i, j in zip(y_predict, y_true):
        if i == 1 and j == 1:
            TP += 1
        elif i == 1 and j == 0:
            FP += 1
        elif i == 0 and j == 0:
            TN += 1
        elif i == 0 and j == 1:
            FN += 1
    return TP, FP, FN, TN 
def ROC(y_prob, y_true, step = 0.1):
    threold = np.arange(0.0, 1.0, step)
    TPR_FPR = []
    for i in threold:
        TP, FP, FN, TN = getMatrix(y_prob > i, y_true)
        TPR_FPR.append([TP / (TP + FN), FP / (TN + FP)])
        #FPR.extend(FP / (TN + FP))
    sorted_ = sorted(TPR_FPR, key=lambda x : x[0])
    return sorted_
def AUC(y_prob, y_true, step = 0.1):
    roc = np.array(ROC(y_prob, y_true, step))
    y = np.cumsum((roc[1:, 0] - roc[:-1, 0]) * roc[:-1, 1])
    return [roc[:-1, 0], y]

def Accurate(yhat, y):
    TP, FP, FN, TN = getMatrix(yhat > 0.5, y)
    return (TP + TN) / (TP + FN + FP + TN)

def logsist(Y):
    return 1 / (1 + torch.exp(Y))

def gaussian_noise(img, mean = 0, sigma = 0.1):
    noise = np.random.normal(mean, sigma, img.shape)
    gau = img + noise
    return gau


# h, w = img.shape[:2]
# center = (w // 2, h // 2)
# # 旋转中心坐标，逆时针旋转：45°，缩放因子：1
# M_1 = cv2.getRotationMatrix2D(center, 45, 1)
# rotated_1 = cv2.warpAffine(img, M_1, (w, h)) 
