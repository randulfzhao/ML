import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重矩阵
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = pd.DataFrame(np.random.randn(self.input_size, self.hidden_size))
        self.W2 = pd.DataFrame(np.random.randn(self.hidden_size, self.output_size))
        
    def sigmoid(self, x):
        # 定义 Sigmoid 函数
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # 定义 Sigmoid 函数的导数
        return x * (1 - x)
    
    def forward(self, X):
        # 前向传播
        self.z = np.dot(X, self.W1)
        self.a = self.sigmoid(self.z)
        self.z2 = np.dot(self.a, self.W2)
        self.output = self.sigmoid(self.z2)
        return self.output
    
    def backward(self, X, y, output):
        # 反向传播
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.z2_error = self.output_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.a)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.a.T.dot(self.output_delta)
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        # 训练模型
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
        # 打印训练误差
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss = {np.mean((y - output) ** 2)}')

    def predict(self, X):
        # 使用已训练的模型预测输出值
        return self.forward(X)


