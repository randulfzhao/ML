'''
在线性回归模型中，我们使用最小二乘法来求解参数 w 和 b。最小二乘法的目标是最小化训练数据的残差平方和，即：

$$J = \sum_{i=1}^{n} (y^{(i)} - (w^T x^{(i)} + b))^2$$

其中 $x^{(i)}$ 和 $y^{(i)}$ 分别表示第 i 个训练样本的特征和目标变量。

在线性回归模型中，我们使用梯度下降法来优化参数 w 和 b，即通过不断更新 w 和 b 的值来最小化损失函数 J。每次更新的步骤如下：

$$w = w - \alpha \frac{\partial J}{\partial w}$$

$$b = b - \alpha \frac{\partial J}{\partial b}$$

其中 $\alpha$ 是学习率，决定了每次更新的步长。梯度下降法的更新过程会一直进行直到参数 w 和 b 收敛，即损失函数 J 达到最小值。

在使用多项式回归时，我们会把特征 $x$ 的各个次方值也作为特征输入模型，这样就可以拟合出更加复杂的非线性模型。

在使用 GLS 时，我们需要对特征 $x$ 进行标准化，即对每一维特征分别减去平均值，再除以标准差。这样做的目的是使每一维的数据具有相同的方差

在使用弹性网络时，我们使用 L1 和 L2 正则化来限制模型的复杂度。L1 正则化会使模型参数的绝对值较小，从而得到稀疏模型，即有很多模型参数的值为 0。L2 正则化会使模型参数的平方和较小，从而得到较小的模型复杂度。


在使用线性回归模型时，需要注意以下几点：
    * 线性回归模型只适用于线性数据。如果数据呈非线性关系，可以使用多项式回归或其他非线性模型。
    * 线性回归模型假设误差是常数的，因此需要对数据进行归一化，以保证所有的数据点在同一数量级，这样才能使得模型的训练更加有效。

我们可以使用如下方式对数据进行归一化：

$$x_{normalized} = \frac{x - \overline{x}}{s}$$

其中 $\overline{x}$ 表示 $x$ 的均值，$s$ 表示 $x$ 的标准差。


评价linear regression 的质量：
    * 均方差 (Mean Squared Error, MSE)：衡量预测值和真实值之间的偏差。MSE 越小越好。

    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2$$

    其中 $y^{(i)}$ 和 $\hat{y}^{(i)}$ 分别表示第 i 个样本的真实值和预测值。

    * 决定系数 (Coefficient of Determination, $R^2$)：衡量预测值和真实值之间的相关性。$R^2$ 越大越好。

    $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{n} (y^{(i)} - \overline{y})^2}$$

    其中 $\overline{y}$ 表示真实值的均值。

    usage:
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

我们还可以使用图像来可视化线性回归模型的效果。例如，我们可以绘制真实值和预测值的散点图，或者绘制预测值的残差（即真实值减去预测值的差值）的直方图，以便更好地理解模型

还有一种常用的方法是使用交叉验证来评估模型的泛化能力。交叉验证的过程如下：
    将训练数据分成 k 个互不重叠的子集。
    使用第 i 个子集作为测试集，剩余的 k-1 个子集作为训练集，训练并评估模型。
    计算 k 次评估的平均值作为模型的最终评估结果。
    交叉验证可以更准确地评估模型的泛化能力，因为它使用了更多的训练数据来训练模型。

'''


import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, fit_intercept=True, method='ols'):
        self.fit_intercept = fit_intercept
        self.method = method
        self.w = None
        self.b = None

    def fit(self, X, y, degree = 1, alpha=1.0, l1_ratio=0.5):
        if self.method == 'polynomial':
            X_poly = np.ones((X.shape[0], 1))
            for i in range(1, degree+1):
                X_poly = np.concatenate((X_poly, X**i), axis=1)
            self.w = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
            return

        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        if self.method == 'ols':
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.method == 'gls':
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X = (X - X_mean) / X_std
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.method == 'ridge':
            self.w = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y
        elif self.method == 'lasso':
            self.w = self._lasso(X, y, alpha)
        elif self.method == 'multi_task_lasso':
            self.w = self._multi_task_lasso(X, y, alpha)
        elif self.method == 'elastic_net':
            n_samples, n_features = X.shape
            identity = np.identity(n_features)
            self.w = np.linalg.inv(X.T @ X + alpha * l1_ratio * identity + alpha * (1 - l1_ratio) * identity) @ X.T @ y
        else:
            raise ValueError('Invalid method')

        if self.fit_intercept:
            self.b = self.w[-1]
            self.w = self.w[:-1]

    def predict(self, X):
        X = self.normalize(X)
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_pred = X @ self.w
        return y_pred

    '''
    usage
    X_test = model.normalize(X_test)
    y_pred = model.predict(X_test)
    '''
    def normalize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    
    def _lasso(self, X, y, alpha):
        # 使用 coordinate descent 算法进行 Lasso 回归
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        for _ in range(100):
            for j in range(n_features):
                X_j = X[:, j]
                w_j = w[j]
                r_j = y - X @ w + w_j * X_j
                w[j] = self._soft_thresholding(X_j, r_j, alpha)
        return w
    
    def _soft_thresholding(self, X, r, alpha):
        if alpha == 0:
            return 0
        elif r.T @ X > alpha:
            return (r.T @ X - alpha) / (X.T @ X)
        elif r.T @ X < -alpha:
            return (r.T @ X + alpha) / (X.T @ X)
        else:
            return 0

    def _multi_task_lasso(self, X, y, alpha):
        # 使用 coordinate descent 算法进行多任务 Lasso 回归
        n_samples, n_features = X.shape
        n_tasks= y.shape[1]
        w = np.zeros((n_tasks, n_features))
        for _ in range(100):
            for j in range(n_features):
                X_j = X[:, j]
            for t in range(n_tasks):
                w_jt = w[t, j]
                r_jt = y[:, t] - X @ w[t, :] + w_jt * X_j
                w[t, j] = self._soft_thresholding(X_j, r_jt, alpha)
        return w

    def _elastic_net(self, X, y, alpha, l1_ratio):
        # 使用 coordinate descent 算法进行 Elastic-Net 回归
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        for _ in range(100):
            for j in range(n_features):
                X_j = X[:, j]
                w_j = w[j]
                r_j = y - X @ w + w_j * X_j
                w[j] = self._soft_thresholding(X_j, r_j, alpha * l1_ratio) + (1 - l1_ratio) * w_j
        return w

# 使用 LinearRegression 类
lr =LinearRegression(fit_intercept=True, method='ols')

# 读取数据
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 进行训练
lr.fit(X, y)

# 预测新数据点的值
X_new = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_pred = lr.predict(X_new)
print(y_pred)

# 使用 Ridge 回归
lr = LinearRegression(fit_intercept=True, method='ridge')
lr.fit(X, y, alpha=1.0)
y_pred = lr.predict(X_new)
print(y_pred)

# 使用 Lasso 回归
lr = LinearRegression(fit_intercept=True, method='lasso')
lr.fit(X, y, alpha=1.0)
y_pred = lr.predict(X_new)
print(y_pred)

# 使用 LinearRegression 类
lr = LinearRegression(fit_intercept=True, method='multi_task_lasso')

# 读取数据
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 进行训练
lr.fit(X, y, alpha=1.0)

# 预测新数据点的值
X_new = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_pred = lr.predict(X_new)
print(y_pred)

# 使用 Elastic-Net
lr = LinearRegression(fit_intercept=True, method='elastic_net')
lr.fit(X, y, alpha=1.0, l1_ratio=0.5)
y_pred = lr.predict(X_new)
print(y_pred)