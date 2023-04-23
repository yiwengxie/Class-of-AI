import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i - 1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def sigmod(self, x): # 掉坑里了，没有负数
        return 1 / (1 + np.exp(-x))

    def dsigmod(self, x):
        return self.sigmod(x) * (1 - self.sigmod(x))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.square(np.tanh(x))

    def forward(self, A):
        for w, b in zip(self.weights, self.biases):
            Z = np.dot(w, A) + b
            A = self.tanh(Z)
        return A

    def backward(self, x, y, learning_rate):
        a_s = [x]
        z_s = []
        for i in range(len(self.layers) - 1):
            z = np.dot(self.weights[i], a_s[-1]) + self.biases[i]
            z_s.append(z)
            a = self.tanh(z)
            a_s.append(a)
        delta = (a_s[-1] - y) * self.dtanh(z_s[-1])
        for i in range(len(self.layers) - 2, -1, -1):
            dw = np.dot(delta, a_s[i].T)
            db = delta
            if i != 0:
                delta = np.dot(self.weights[i].T, delta) * self.dtanh(z_s[i - 1])
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, Y, epochs, learning_rate):
        for epochs in range(epochs):
            print(f"epochs: {epochs}")
            for i in range(len(X)):
                x = X[i].reshape(-1, 1)
                y = Y[i].reshape(-1, 1)
                self.backward(x, y, learning_rate)


# 生成随机数据
X_train = np.linspace(-np.pi, np.pi, 1000)
X_test = np.linspace(-np.pi, np.pi, 1000)
Y_train = np.sin(X_train)
Y_test = np.sin(X_test)

model = Network([1, 20, 20, 1])

# 使用反向传播算法进行回归拟合
model.train(X_train, Y_train, learning_rate=0.01, epochs=3000)

result = [float(model.forward(i)) for i in X_test]
result = np.array(result)
print(f"average error: {np.mean(abs(Y_test-result))}")


# 绘制拟合曲线和真实曲线
plt.plot(X_train, Y_train, label='sin(x)')
plt.plot(X_test, result, label='fitting curve')
plt.legend()
plt.show()
