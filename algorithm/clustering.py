'''

to enhance porformance:

* method to compute distance: adopt 曼哈顿距离、切比雪夫距离

* regulation for the maximum iteration times

* use random k centers as center of clustering可以通过更新聚类中心的方差和均值来计算新的聚类中心。这样可以大大减少计算量，提高计算效率。

* evaluation method: 轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数

'''


import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k, max_iter=100, distance='euclidean'):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroids = None
        self.closest_centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # 初始化聚类中心
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        for i in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            if self.distance == 'euclidean':
                distances = np.array([np.linalg.norm(x - self.centroids, axis=1) for x in X])
            elif self.distance == 'manhattan':
                distances = np.array([np.abs(x - self.centroids).sum(axis=1) for x in X])
            # 找到每个点最近的聚类中心
            self.closest_centroids = np.argmin(distances, axis=1)
            # 计算新的聚类中心
            new_centroids = np.array([X[self.closest_centroids == i].mean(axis=0) for i in range(self.k)])
            # 如果聚类中心没有变化，退出循环
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        # 计算每个点到聚类中心的距离
        if self.distance == 'euclidean':
            distances = np.array([np.linalg.norm(x - self.centroids, axis=1) for x in X])
        elif self.distance == 'manhattan':
            distances = np.array([np.abs(x - self.centroids).sum(axis=1) for x in X])
        # 找到每个点最近的聚类中心，并返回聚类结果
        return np.argmin(distances, axis=1)

# 使用 KMeans 类
kmeans = KMeans(k=3, max_iter=50, distance='manhattan')

# 读取数据
df = pd.read_csv('data.csv')
X = df.values

# 进行聚类
kmeans.fit(X)

# 输出聚类中心
print(kmeans.centroids)

# 预测新数据点的聚类结果
X_new = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_pred = kmeans.predict(X_new)
print(y_pred)
