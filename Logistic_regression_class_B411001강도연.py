import numpy as np
import matplotlib.pyplot as plt

class Logistic_regression:
    def __init__(self):
        self.train_set = 0
        self.train_target = 0
        self.weight = 0
        self.class_num = 0                                                                                              # binary로 학습시, 학습할 class
        self.case = 0                                                                                                   # case가 0 = binary class, 1 = multiclass


    def cost(self, y, h, case):                                                                                         # cost(target, hypothesis, 0(binary) or 1(multi))
        epsilon = 1e-150                                                                                                # cost값이 nan이 나오는것을 방지 하기 위한 작은값
        if case == 0:                                                                                                   # binary로 학습시
            return np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)) / (-1 * y.shape[0])
        else:                                                                                                           # multiclass로 학습시, class별로 cost값 계산
            return (y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)).sum(axis=0) / (-1 * y.shape[0])


    def learn(self, x, y, rate, epoch, case, class_num=None):                                                           # learn(train_data, train_target, rate, epoch, 0 or 1, class to learn)
        self.train_set = x
        self.train_target = y

        def add_bias(x):                                                                                                # input feature들에 bias 값 1을 더하기 위한 function
            bias = np.full((len(x), 1), 1)
            return np.concatenate((bias, x), axis=1)

        def y_trans(x, y):                                                                                              # binary로 학습 위해 target class에 해당하는 target은 1 나머지는 0으로 reshape
            for j in range(len(x)):                                                                                     # y_trans(data, target)
                if x[j] != y:
                    x[j] = 0
                else:
                    x[j] = 1
            return np.array(x).reshape(len(x), 1)

        def one_hot_encoding(y):                                                                                        # multiclass로 학습 위해 output을 one hot encoding
            temp = np.unique(y, axis=0)                                                                                 # one_hot_encoding(target)
            temp = temp.shape[0]
            return np.eye(temp)[y]

        def sigmoid(z):                                                                                                 # sigmoid 함수
            return 1 / (1 + np.exp(-z))

        self.case = case                                                                                                # case = 0 이면 binary로 학습, case = 1 이면 multiclass로 학습
        self.class_num = class_num                                                                                      # binary로 학습시 학습할 class
        train_set = add_bias(x)                                                                                         # train_data에 bias 추가

        if case == 0:                                                                                                   # binary로 학습시 y_trans
            train_target = y_trans(y, class_num)
        else:
            train_target = one_hot_encoding(y)                                                                          # multi로 학습시 one_hot_encoding

        self.weight = np.random.randn(train_set.shape[1], train_target.shape[1])                                        # data feature 개수, class 개수 에 맞춰서 가중치 초기화

        cost = []                                                                                                       # plotting을 위해 cost값 저장 용도
        itera = []                                                                                                      # plotting을 위해 반복횟수 저장 용도
        for j in range(epoch):                                                                                          # epoch 만큼 학습
            itera.append(j + 1)
            z = np.dot(train_set, self.weight)
            a = sigmoid(z)                                                                                              # activation함수 적용
            cost_temp = self.cost(train_target, a, case)                                                                # cost값 계산
            cost.append(cost_temp)
            print("epoch:", j, "       cost:", cost_temp)
            temp = []
            for i in range(train_set.shape[1]):                                                                         # feature 수 만큼
                if case == 0:                                                                                           # binary 학습시
                    r = np.sum((a - train_target) * train_set[:, i].reshape(train_set.shape[0], 1))
                else:
                    r = ((a - train_target) * train_set[:, i].reshape(train_set.shape[0], 1)).sum(axis=0)               # multiclass 학습시 class별로 theta값 계산위해 .sum(axis=0) 적용
                temp.append(r)
            sum1 = np.array(temp).reshape(len(temp), train_target.shape[1])
            self.weight = self.weight - (rate * sum1)                                                                   # 개선된 weight 값 계산 및 저장
        plt.plot(itera, cost)                                                                                           # cost값과 iteration 수로 plotting
        plt.show()


    def predict(self, x, y):                                                                                            # predict(test_data, test_data_target)
        def add_bias(x):
            bias = np.full((len(x), 1), 1)
            return np.concatenate((bias, x), axis=1)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        test_set = add_bias(x)
        z = np.dot(test_set, self.weight)
        count = 0                                                                                                       # 정확도 계산 위한 count
        a = sigmoid(z)                                                                                                  # 학습 후 얻은 weight값을 이용한 후 activation 적용
        if self.case == 0:                                                                                              # binary로 학습한 경우 accuracy 계산
            result = a > 0.5                                                                                            # activation적용한 값이 > 0.5 인것들로 거른다.
            for i in range(len(y)):
                if self.class_num == y[i]:                                                                              # 학습한 target_class와 test_data_target가 같고 result가 true(>0.5)면 예측성공
                    if result[i]:
                        count += 1
                else:
                    if not result[i]:                                                                                   # 학습한 target_class와 test_data_target가 다르고 result가 false(<0.5)면 예측성공
                        count += 1
        else:                                                                                                           # multiclass로 학습한 경우 accuracy 계산
            result = np.argmax(a, axis=1)                                                                               # activation 적용한 결과에서, 값이 가장 높은 값의 class가 output (index가 class)
            for j in range(len(y)):
                if y[j] == result[j]:
                    count += 1

        print()
        return print("accuracy:", count / len(y))                                                                       # accuracy 출력




