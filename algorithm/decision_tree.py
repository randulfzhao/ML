'''
如何选择最优的划分特征：可以使用信息增益或基尼系数等指标来选择划分特征。

如何处理缺失值：可以使用平均值或中位数来填补缺失值，或者将缺失值看作一种特殊的值。

如何进行剪枝：在训练过程中，可以使用验证集来评估模型的性能，并根据验证集的表现决定是否剪枝。

如何处理连续值：可以将连续值离散化，或者使用均值划分等方法来处理。
'''


import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, random_state=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree = None
        self.trees = None

    def predict(self, X):
        y_pred = []
        for sample in X:
            y_pred.append(self._predict_sample(sample))
        return y_pred
        
    def _grow_tree(self, X, y, depth=0):
        # 确定终止条件
        if self._terminate(y, depth):
            return self._predict(y)
        
        # 选择最优划分特征
        best_feature = self._best_split(X, y)
        
        # 创建决策树
        tree = {best_feature: {}}
        unique_values = np.unique(X[:, best_feature])
        for value in unique_values:
            mask = X[:, best_feature] == value
            sub_X = X[mask]
            sub_y = y[mask]
            subtree = self._grow_tree(sub_X, sub_y, depth+1)
            tree[best_feature][value] = subtree
        
        return tree

    def _terminate(self, y):
        # 如果所有样本属于同一类别，则终止划分
        if len(set(y)) == 1:
            return True
        
        # 如果树的深度超过最大深度，则终止划分
        if self.depth >= self.max_depth:
            return True
        
        # 如果分裂后的子节点样本数量小于最小样本数量限制，则终止划分
        if len(y) < self.min_samples_split:
            return True
        
        return False


    def _predict_sample(self, sample):
        tree = self.tree
        while isinstance(tree, dict):
            feature, value = list(tree.items())[0]
            tree = tree[feature][sample[feature]]
        return tree
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.tree = self._grow_tree(X, y)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        best_gain = 0
        entropy = self._entropy(y)
        for feature in range(n_features):
            values = X[:, feature]
            unique_values = np.unique(values)
            for value in unique_values:
                X_sub, y_sub = self._split(X, y, feature, value)
                sub_entropy = self._entropy(y_sub)
                gain = entropy - sub_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
        return best_feature

    def _information_gain(self, y, values):
        # 计算信息增益
        total_entropy = self._entropy(y)
        unique_values, counts = np.unique(values, return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, counts):
            mask = values == value
            sub_y = y[mask]
            entropy = self._entropy(sub_y)
            weighted_entropy += count/len(y) * entropy
        return total_entropy - weighted_entropy
    

    def _predict(self, y):
        # 返回出现次数最多的类别
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        y_pred = []
        for sample in X:
            y_pred.append(self._predict_sample(sample))
        return y_pred


    '''
        熵是一种度量信息级别的指标，在决策树模型中常用于计算信息增益。
        通常来说，熵越大，则信息越复杂；熵越小，则信息越简单。
        例如，在二分类问题中，如果 y 中的标签分布较为均匀，则熵值较大；
        如果 y 中的标签分布较为不均匀，则熵值较小。
        
        计算 y 中每个标签出现的次数，并计算每个标签在 y 中出现的概率。
        使用概率计算 y 的熵，并返回熵值。
    '''
    def _entropy(self, y):
        # 计算熵
        unique_labels, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        entropy = -sum(p * np.log2(p) for p in prob)
        return entropy

    def entropy(y):
        # 计算熵
        unique_labels, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        entropy = -sum(p * np.log2(p) for p in prob)
        return entropy

    def _most_common_label(self, y):
        # 返回 y 中出现次数最多的标签
        unique_labels, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        max_index = counts.argmax()
        return unique_labels[max_index]
    
    def _predict_sample(self, sample):
        # 预测单个样本的标签
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            value = sample[feature]
            if value in node[feature]:
                node = node[feature][value]
            else:
                return self._most_common_label(self.y)
        return node
    
    def _prune(self, X, y, tree):
        # 如果当前节点是叶节点，则返回
        if not isinstance(tree, dict):
            return tree
        
        # 递归剪枝子树
        feature, subtree = list(tree.items())[0]
        for value in subtree.values():
            X_sub, y_sub = self._split(X, y, feature, value)
            subtree[value] = self._prune(X_sub, y_sub, subtree[value])
        
        # 如果剪枝后的子树均为叶节点，则将

    def prune(self, X_val, y_val):
        self.tree = self._prune(X_val, y_val, self.tree)




    def fit_random_forest(self, X, y, n_trees=10):
        # 训练多棵决策树，并将其组合成随机森林
        trees = []
        for _ in range(n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, random_state=self.random_state)
            tree.fit(X, y)
            trees.append(tree)
        self.trees = trees

    def predict_random_forest(self, X):
        y_pred = []
        for sample in X:
            y_sub_pred = []
            for tree in self.trees:
                y_sub_pred.append(tree._predict_sample(sample))
            y_pred.append(self._predict(y_sub_pred))
        return y_pred

    """
    Bagging是一种基于多个决策树的集成方法，其中每棵决策树之间是独立的，通常使用随机森林就是使用bagging的方法。

    你可以在决策树类中加入一个函数，用来训练多棵决策树，然后将这些决策树的预测结果进行投票（对于分类问题）或平均（对于回归问题），得到最终的结果。
    
    Train n_estimators decision trees using bagging.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        Training data.
    y : array-like or pd.Series, shape (n_samples,)
        Target values.
    n_estimators : int, default=10
        Number of decision trees to train.
    max_samples : int or float, default=None
        The number of samples to draw from X to train each decision tree.
            If int, then draw max_samples samples.
            If float, then draw max_samples * X.shape[0] samples.
    max_features : int or float, default=None
        The number of features to consider when looking for the best split.
            If int, then consider max_features features.
            If float, then consider max_features * X.shape[1] features.
    random_state : int, default=None
        Random seed for sampling and tree induction.
        
    Returns
    -------
    self : DecisionTree
        Return self.
    """
    def bagging(self, X, y, n_estimators=10, max_samples=None, max_features=None, random_state=None):
        self.estimators_ = []
        for i in range(n_estimators):
            # Sample the training set
            if isinstance(max_samples, int):
                samples = np.random.choice(X.shape[0], max_samples, replace=True)
            elif isinstance(max_samples, float):
                samples = np.random.choice(X.shape[0], int(max_samples * X.shape[0]), replace=True)
            else:
                samples = np.arange(X.shape[0])
            X_samples = X.iloc[samples]
            y_samples = y.iloc[samples]
            # Train a decision tree
            tree = DecisionTree(max_features=max_features, random_state=random_state)
            tree.fit(X_samples, y_samples)
            self.estimators_.append(tree)
        return self

    """
    Boosting是一种基于单个决策树的集成方法，其中每棵决策树之间是有序的，通常使用提升树就是使用boosting的方法。

    你可以在决策树类中加入一个函数，用来训练多棵决策树，每棵决策树的训练是基于之前决策树的错误率的。

    Train n_estimators decision trees using boosting.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        Training data.
    y : array-like or pd.Series, shape (n_samples,)
        Target values.
    n_estimators : int, default=10
        Number of decision trees to train.
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree by learning_rate.
    random_state : int, default=None
        Random seed for tree induction.
        
    Returns
    -------
    self : DecisionTree
        Return self.
    """
    def boosting(self, X, y, n_estimators=10, learning_rate=0.1, random_state=None):
        self.estimators_ = []
        self.learning_rate_ = learning_rate
        # Initialize the prediction to the average of target values
        y_pred = np.full(y.shape, np.mean(y))
        for i in range(n_estimators):
            # Calculate the residuals
            residuals = y - y_pred
            # Train a decision tree on the residuals
            tree = DecisionTree(random_state=random_state)
            tree.fit(X, residuals)
            self.estimators_.append(tree)
            # Update the prediction
            y_pred += learning_rate * tree.predict(X)
        return self
