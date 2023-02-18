import numpy as np
import pandas as pd

'''
y_true: 一个 numpy 数组或 pandas Series，表示真实的标签。
y_pred: 一个 numpy 数组或 pandas Series，表示预测的标签。
'''

def mean_squared_error(y_true, y_pred):
    # 计算差的平方
    squared_error = np.square(y_true - y_pred)
    
    # 计算均值
    mse = np.mean(squared_error)
    
    # 返回 MSE
    return mse

def cross_entropy_loss(y_true, y_pred):
    # 计算交叉熵损失
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
def mean_absolute_error(y_true, y_pred):
    # 计算平均绝对误差
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    # 计算 y 的均值
    y_mean = np.mean(y_true)
    
    # 计算 y_true 与 y_mean 的差的平方和
    sum_of_squared_residuals = np.sum(np.square(y_true - y_mean))
    
    # 计算 y_pred 与 y_mean 的差的平方和
    sum_of_squared_prediction_error = np.sum(np.square(y_pred - y_mean))
    
    # 计算 r2 score
    r2 = 1 - (sum_of_squared_prediction_error / sum_of_squared_residuals)
    
    # 返回 r2 score
    return r2

'''
model: 一个模型对象，包含 fit 和 predict 方法。
X: 一个 numpy 数组或 pandas DataFrame，表示训练数据。
y: 一个 numpy 数组或 pandas Series，表示标签。
k: 一个整数，表示 k-fold 交叉验证的 k 值。

使用 numpy 的 linspace 函数和 arange 函数，生成一组序列，表示数据集的下标。
然后，使用 numpy 的 shuffle 函数打乱这些下标，以便于每次进行 k-fold 交叉验证时选择不同的数据。
接着，你需要使用 numpy 的 split 函数，将序列划分为 k 个分区。
然后，使用循环遍历每个分区，并将当前分区作为测试集，其余分区作为训练集。使用 model 的 fit 和 predict 方法训练模型并预测，计算 r2 score 和 MSE。
最后，使用 numpy 的 mean 函数计算所有 r2 score 和 MSE 的均值，并返回结果。
'''

def kfold(model, X, y, k):
    # 生成数据集的下标序列
    indices = np.arange(len(X))
    
    # 打乱下标序列
    np.random.shuffle(indices)
    
    # 将下标序列划分为 k 个分区
    folds = np.split(indices, k)
    
    # 初始化 r2 score 和 MSE 的列表
    r2_scores = []
    mses = []
    
    # 遍历每个分区
    for fold in folds:
        # 将当前分区作为测试集，其余分区作为训练集
        X_train, X_test = X[fold], X[np.setdiff1d(indices, fold)]
        y_train, y_test = y[fold], y[np.setdiff1d(indices, fold)]
        
        # 训练模型并预测
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 计算 r2 score 和 MSE
        r2_scores.append(r2_score(y_test, y_pred))
        mses.append(mean_squared_error(y_test, y_pred))
    
    # 计算 r2 score 和 MSE 的均值
    r2_mean = np.mean(r2_scores)
    mse_mean = np.mean(mses)
    
    # 返回结果
    return r2_mean, mse_mean