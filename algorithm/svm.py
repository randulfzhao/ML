import cvxopt
import numpy as np
import pandas as pd

class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.w = None
        self.b = None

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape

        # 计算 Gram 矩阵
        if self.kernel == 'linear':
            K = X @ X.T
        elif self.kernel == 'poly':
            K = (X @ X.T + 1) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'auto':
                self.gamma = 1 / n_features
            K = np.exp(-self.gamma * np.square(X[:, np.newaxis] - X))
        else:
            raise ValueError('Invalid kernel')

        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 计算 Gram 矩阵
        K = self._gram_matrix(X)

        # 求解二次规划问题
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples))))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # 获取支持向量
        self.w = np.array(sol['x'])
        support_vectors = self.w > 1e-5
        self.b = y[support_vectors] - np.dot(X[support_vectors], self.w)

    def predict(self, X):
        # 计算 Gram 矩阵
        K = self._gram_matrix(X)
        return K @ self.w + self.b

# 使用 SVM 类
svm = SVM(C=1.0, kernel='linear')

# # 读取数据
# df = pd.read_csv('data.csv')
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # 进行训练
# svm.fit(X, y)

# # 预测新数据点的类别
# X_new = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# y_pred = svm.predict(X_new)
# print(y_pred)