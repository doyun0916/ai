import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(1, hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(1, output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.x = 0
        self.t = 0

    def predict(self, x):
        def sigmoid(z):  # sigmoid 함수
            return 1 / (1 + np.exp(-z))

        def softmax(x):
            exp_a = np.exp(x)
            sum_exp_a = np.sum(exp_a, axis=1).reshape(x.shape[0], 1)
            return exp_a / sum_exp_a

        z2 = np.dot(x, self.params['W1']) + self.params['b1']
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.params['W2']) + self.params['b2']
        y = softmax(z3)
        return y

    def loss(self, x, t):
        def cross_entropy_error(y, t):                                                       #cross_enthropy_error (one_hot_encoding)
            epsilon = 1e-7
            if y.ndim == 1:  # 2차원으로 바꿔줌
                t = t.reshape(1, t.size)
                y = y.reshape(1, y.size)
            batch_size = y.shape[0]
            cee = np.sum(-t * np.log10(y + epsilon)) / batch_size
            return cee

        output = self.predict(x)
        return cross_entropy_error(output, t)

    def accuracy(self, x, t):
        accuracy = 0
        result = self.predict(x)
        output = np.argmax(result, axis=1)
        t_labeled = np.argmax(t, axis=1)
        for i in range(len(t_labeled)):
            if output[i] == t_labeled[i]:
                accuracy += 1
        accuracy_final = accuracy / len(t_labeled)
        return accuracy_final

    def numerical_gradient(self, x, t):
        init = self.params

        def gradient(f, w):
            h = 1e-4
            grad = np.zeros_like(w)
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    wi = w[i][j]
                    w[i][j] = wi + h
                    fx1 = f(x, t)
                    w[i][j] = wi - h
                    fx2 = f(x, t)
                    grad[i][j] = (fx1 - fx2) / (2 * h)
                    w[i][j] = wi
            self.params = init
            return grad

        w1_new = gradient(self.loss, self.params['W1'])
        b1_new = gradient(self.loss, self.params['b1'])
        w2_new = gradient(self.loss, self.params['W2'])
        b2_new = gradient(self.loss, self.params['b2'])

        return w1_new, b1_new, w2_new, b2_new



    def learn(self, x, t, lr, epoch, size):

        def one_hot_encoding(y):  # multiclass로 학습 위해 output을 one hot encoding
            temp = np.unique(y, axis=0)  # one_hot_encoding(target)
            temp = temp.shape[0]
            return np.eye(temp)[y]

        one_hot_encoded_t = one_hot_encoding(t)

        self.x = x
        self.t = one_hot_encoded_t


        # gradient descent 부분 함수 지우고 그냥 이것만 빼서 써! 너가 나중에 참고하라고 아직 안뺌
        init = ['W1', 'b1', 'W2', 'b2']
        epoch_count = []
        loss_count = []
        accuracy_count = []
        for i in range(epoch):
            batch_size = min(size, self.x.shape[0])                              # 미니배치
            batch_mask = np.random.choice(self.x.shape[0], batch_size)
            x_batch = self.x[batch_mask]
            t_batch = self.t[batch_mask]

            cal = self.numerical_gradient(x_batch, t_batch)
            for j in range(4):                                          # 4번해야지 cause w1, b1, w2, b2
                self.params[init[j]] -= lr * cal[j]

            loss = self.loss(x_batch, t_batch)
            accuracy = self.accuracy(x_batch, t_batch)
            epoch_count.append(i)  # 그래프 위한 저장
            loss_count.append(loss)
            accuracy_count.append(accuracy)
            print("cost:", loss, "accuracy", accuracy)  # 출력

        plt.plot(epoch_count, loss_count)  # cost값과 iteration 수로 plotting
        plt.plot(epoch_count, accuracy_count)  # accuracy값과 iteration 수로 plotting
        plt.show()
        return None


        # epoch마다 cost,accuracy 뽑고, 마지막에 graph까지

        