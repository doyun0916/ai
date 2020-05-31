import numpy as np
import matplotlib.pyplot as plt                                                                                         #graph plotting을 위한 library


class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):   # init() --> 받은 3개의 size들을 통해 Weight 및 bias 초기화
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(1, hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(1, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.x = 0                                                                                                      # train data 저장하기 위한 변수
        self.t = 0                                                                                                      # train data target 저장하기 위한 변수

    def predict(self, x):                                       # 받은 input data가 class별로 속할 확률 계산
        def sigmoid(z):                                                                                                 # sigmoid 함수
            return 1 / (1 + np.exp(-z))

        def softmax(x):                                                                                                 # 여러 data를 처리할 수 있게 끔 수정된 softmax함수
            exp_a = np.exp(x)
            sum_exp_a = np.sum(exp_a, axis=1).reshape(x.shape[0], 1)                                                    # data별로 계산을 할 수 있도록 각 행마다 sum을 해준다.
            return exp_a / sum_exp_a

        z2 = np.dot(x, self.params['W1']) + self.params['b1']                                                           # Layer1(feature) 값에 weight와 bias를 적용한다.
        a2 = sigmoid(z2)                                                                                                # 위 결과에 sigmoid에 적용한다.
        z3 = np.dot(a2, self.params['W2']) + self.params['b2']                                                          # Layer2 값에 layer2 weight와 bias를 적용한다.
        y = softmax(z3)                                                                                                 # 위 결과에 softmax를 적용
        return y

    def loss(self, x, t):                                       # cross-entropy를 이용하여 data와 one-hot-encoding된 data target에 따른 loss 계산
        def cross_entropy_error(y, t):                                                                                  # cross_entropy_error 함수
            epsilon = 1e-7
            if y.ndim == 1:                                                                                             # data가 1차원이면, 2차원으로 바꿔줌
                t = t.reshape(1, t.size)
                y = y.reshape(1, y.size)
            batch_size = y.shape[0]
            cee = np.sum(-t * np.log10(y + epsilon)) / batch_size                                                       # data가 여러개 들어올 경우, 그만큼 나누어 준다.
            return cee

        output = self.predict(x)                                                                                        # data가 network를 통하여 softmax가 적용된 output
        return cross_entropy_error(output, t)                                                                           # 위 값과, 실제 target값을 이용한 loss값 계산

    def accuracy(self, x, t):                                   # data와 one-hot-encoding된 data target을 받아 계산된 결과에 따른 정확도 계산
        accuracy = 0
        result = self.predict(x)                                                                                        # data가 network를 통하여 softmax가 적용된 output
        output = np.argmax(result, axis=1)                                                                              # 위 값중 가장 큰 값을 가진 class선택
        t_labeled = np.argmax(t, axis=1)                                                                                # output과 비교 위해 one_hot_encoding 다시 label로
        for i in range(len(t_labeled)):                                                                                 # 계산된 class와 실제 target 비교를 통한 accuracy계산
            if output[i] == t_labeled[i]:
                accuracy += 1
        accuracy_final = accuracy / len(t_labeled)
        return accuracy_final

    def numerical_gradient(self, x, t):                         # data와 one_hot_encoding된 data target이 적용된 loss function에서 각 Weight, bias값들의 편미분 값 계산
        def gradient(f, w):
            h = 1e-4
            grad = np.zeros_like(w)                                                                                     # 계산된 gradient 저장 위해 빈 np.array 초기화
            for i in range(w.shape[0]):                                                                                 # 각각 weight에 대해 편미분(기울기)계산을 위해 행 별로
                for j in range(w.shape[1]):                                                                             # 열 만큼 반복
                    wi = w[i][j]
                    w[i][j] = wi + h
                    fx1 = f(x, t)                                                                                       # loss function에 weight + h를 넣은 결과값
                    w[i][j] = wi - h
                    fx2 = f(x, t)                                                                                       # loss function에 weight - h를 넣은 결과값
                    grad[i][j] = (fx1 - fx2) / (2 * h)                                                                  # 중앙차분을 이용한 gradient 계산 후 저장
                    w[i][j] = wi                                                                                        # 다음 값 계산을 위해 원래 값으로 돌려준다.
            return grad

        w1_new = gradient(self.loss, self.params['W1'])                                                                 # W1에서 각 값들에 대한 gradient가 계산된 np.array
        b1_new = gradient(self.loss, self.params['b1'])                                                                 # b1에서 각 값들에 대한 gradient가 계산된 np.array
        w2_new = gradient(self.loss, self.params['W2'])                                                                 # W2에서 각 값들에 대한 gradient가 계산된 np.array
        b2_new = gradient(self.loss, self.params['b2'])                                                                 # b2에서 각 값들에 대한 gradient가 계산된 np.array
        return w1_new, b1_new, w2_new, b2_new

    def learn(self, x, t, lr, epoch, size):    # W1, b1, W2,b2의 편미분, learning rate를 가지고 gradient descent를 적용하여 params의 값을 수정, 이 과정을 epoch만큼 반복
        def one_hot_encoding(y):                                                                                        # multiclass로 학습 위해 target값을 one hot encoding
            temp = np.unique(y, axis=0)
            temp = temp.shape[0]
            return np.eye(temp)[y]

        one_hot_encoded_t = one_hot_encoding(t)

        self.x = x                                                                                                      # input data 저장
        self.t = one_hot_encoded_t                                                                                      # one_hot_encoding된 input data target 저장
        init = ['W1', 'b1', 'W2', 'b2']                                                                                 # 계산의 편의를 위해 생성
        epoch_count = []                                                                                                # graph plotting을 위해 각 epoch 저장할 변수
        loss_count = []                                                                                                 # graph plotting을 위해 각 loss 저장할 변수
        accuracy_count = []                                                                                             # graph plotting을 위해 각 accuracy 저장할 변수
        for i in range(epoch):                                                                                          # 정해진 epoch만큼 수행
            batch_size = min(size, self.x.shape[0])                                                                     # mini-batch시 input data 크기 넘기지 않도록 설정
            batch_mask = np.random.choice(self.x.shape[0], batch_size)                                                  # random하게 batch_size 만큼 선택
            x_batch = self.x[batch_mask]
            t_batch = self.t[batch_mask]

            cal = self.numerical_gradient(x_batch, t_batch)                                                             # 선택된 batch data들을 이용한 W,b들의 편미분 계산
            for j in range(4):                                                                                          # 위 값을 이용하여 W1,b1,W2,b2에 대한,
                self.params[init[j]] -= lr * cal[j]                                                                     # gradient descent 적용 및 바뀐 params 값 저장

            loss = self.loss(x_batch, t_batch)                                                                          # 수정된 params을 이용한 loss값 계산
            accuracy = self.accuracy(x_batch, t_batch)                                                                  # 수정된 params을 이용한 accuracy값 계산
            epoch_count.append(i)                                                                                       # 그래프 plotting 위한 epoch값 저장
            loss_count.append(loss)                                                                                     # 그래프 plotting 위한 loss값 저장
            accuracy_count.append(accuracy)                                                                             # 그래프 plotting 위한 accuracy값 저장
            print("cost:", loss, "      accuracy", accuracy)                                                            # epoch마다 cost, accuracy 출력

        plt.plot(epoch_count, loss_count, label="loss")                                                                 # 각 epoch 당 loss값을 이용하여 plotting
        plt.plot(epoch_count, accuracy_count, label="accuracy")                                                         # 각 epoch 당 accuracy값을 이용하여 plotting
        plt.legend()
        plt.show()                                                                                                      # graph plot
        return None