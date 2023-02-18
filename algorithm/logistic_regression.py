import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, fit_intercept=True, solver='gd'):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_pred):
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def fit(self, X, y, learning_rate=0.01, n_iter=100, tol=1e-6):
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        
        self.w = np.zeros(X.shape[1])
        if self.solver == 'gd':
            for _ in range(n_iter):
                z = X @ self.w
                y_pred = self.sigmoid(z)
                grad = X.T @ (y_pred - y)
                self.w -= learning_rate * grad
        elif self.solver == 'newton':
            for _ in range(n_iter):
                z = X @ self.w
                y_pred = self.sigmoid(z)
                hessian = X.T @ np.diag(y_pred * (1 - y_pred)) @ X
                grad = X.T @ (y_pred - y)
                delta = np.linalg.inv(hessian) @ grad
                self.w -= delta
                if np.linalg.norm(delta) < tol:
                    break
        else:
            raise ValueError('Invalid solver')

        if self.fit_intercept:
            self.b = self.w[-1]
            self.w = self.w[:-1]

    def predict_proba(self, X):
        if self.fit_intercept:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        z = X @ self.w
        y_proba = self.sigmoid(z)
        return y_proba

    def predict(self, X, threshold=0.5):
        y_proba = self.predict_proba(X)
        y_pred = (y_proba > threshold).astype(int)
        return y_pred

# 使用 LogisticRegression 类
lr = LogisticRegression(fit_intercept=True, solver='newton')
