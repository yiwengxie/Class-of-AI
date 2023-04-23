import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


class Preprocess:
    def __init__(self, path):
        X = []  # 定义图像名称
        Y = []  # 定义图像标签
        for i in range(1, 13):
            imgList = os.listdir(path + "/%s" % i)
            # imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按顺序读如图片
            for name in imgList:
                X.append(path + "/" + str(i) + "/" + str(name))
                vector = np.zeros(12)
                vector[i - 1] = 1
                Y.append(vector)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def split(self, test_size: float):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=2)
        XX_train = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in X_train]
        XX_test = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in X_test]
        XX_train = np.array(XX_train) / 255.0
        XX_test = np.array(XX_test) / 255.0
        return XX_train, XX_test, Y_train, Y_test

    def testread(self):
        X_test = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in self.X]
        X_test = np.array(X_test) / 255.0
        return X_test, self.Y


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i - 1]) / np.sqrt(layers[i - 1]))
            self.biases.append(np.random.randn(layers[i], 1) / np.sqrt(layers[i]))

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmod(self, x):
        return self.sigmod(x) * (1 - self.sigmod(x))

    def leakyrelu(self, x):
        return np.maximum(0.01 * x, x)

    def dleakyrelu(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = 0.01
        return dx

    def forward(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.leakyrelu(z)
        return a

    def backward(self, x, y, learning_rate, accuracy):
        a_s = [x]
        z_s = []
        for i in range(len(self.layers) - 1):
            z = np.dot(self.weights[i], a_s[-1]) + self.biases[i]
            z_s.append(z)
            a = self.leakyrelu(z)
            a_s.append(a)
        if np.argmax(a_s[-1]) == np.argmax(y):
            accuracy[0] += 1
        delta = (a_s[-1] - y) * self.dleakyrelu(z_s[-1])
        for i in range(len(self.layers) - 2, -1, -1):
            dw = np.dot(delta, a_s[i].T)
            db = delta
            if i != 0:  # 坑死爹了
                delta = np.dot(self.weights[i].T, delta) * self.dleakyrelu(z_s[i - 1])  # 是z，是i （坑死老子了）
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, Y, epochs, learning_rate=0.01, draw='no'):

        fig, ax = plt.subplots()
        plt.ion()
        x_data = []
        y1_data = []
        y2_data = []
        for epoch in range(epochs):
            print(f"epoch={epoch + 1} is running")
            accuracy = [0]
            for i in range(len(X)):
                x = X[i].reshape(-1, 1)
                y = Y[i].reshape(-1, 1)
                self.backward(x, y, learning_rate, accuracy)

            # 画画
            if draw != 'no':
                x_data.append(epoch)
                y1_data.append(self.predict_for_train(X_test, Y_test))
                y2_data.append(accuracy[0] / len(X))
                ax.clear()
                ax.plot(x_data, y1_data, y2_data)
                plt.title("Accuracy Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.pause(0.01)

            if epoch == 30:
                with open('model_weights5.pkl', 'wb') as f:
                    pickle.dump(self.weights, f)
                with open('model_biases5.pkl', 'wb') as f:
                    pickle.dump(self.biases, f)

    def predict_for_train(self, X, Y):
        accuracy = 0
        for i, j in zip(X, Y):
            prediction = self.forward(i.reshape(-1, 1))
            result = np.argmax(prediction) + 1
            actual = np.argmax(j) + 1
            if result == actual:
                accuracy += 1
            # print(f"Prediction: {result},  Actual：{actual}")
        print(f"Accuracy: {accuracy / len(Y)}")
        return accuracy / len(Y)

    def predict(self, X, Y):
        with open('model_weights5.pkl', 'rb') as f:
            self.weights = pickle.load(f)
        with open('model_biases5.pkl', 'rb') as f:
            self.biases = pickle.load(f)
        accuracy = 0
        for i, j in zip(X, Y):
            prediction = self.forward(i.reshape(-1, 1))
            result = np.argmax(prediction) + 1
            actual = np.argmax(j) + 1
            if result == actual:
                accuracy += 1
            # print(f"Prediction: {result},  Actual：{actual}")
        print(f"Accuracy: {accuracy / len(Y)}")
        return accuracy / len(Y)


if __name__ == "__main__":
    preprocess = Preprocess('/Volumes/LCZ/test_data')
    model = Network([784, 256, 128, 12])
    # for train
    # X_train, X_test, Y_train, Y_test = preprocess.split(0.3)
    # model.train(X_train, Y_train, learning_rate=0.005, epochs=100, draw="yes")

    # for test
    X_test, Y_test = preprocess.testread()
    model.predict(X_test, Y_test)
