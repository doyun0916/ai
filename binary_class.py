import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()                                                   # iris data load
x = iris.data                                                        # iris data with features
y = iris.target                                                      # iris data's 실제 output
iris_name = iris.target_names                                        # iris 3종류의 이름
test, test_target_temp, train, train_target_temp = [], [], [], []              # test, test target = test 데이터셋, output 저장
for i in range(x.shape[0]):                                          # train, train_target = train 데이터셋, output 저장
    if i % 15 == 14:                 # test 데이터셋과 output, train 데이터셋과 output 분리
        test.append([x[i]])
        test_target_temp.append(y[i])
    else:
        train.append([x[i]])
        train_target_temp.append(y[i])

def add_bias(x):
    bias = np.full((len(x), 1), 1)
    temp = np.array(x).reshape(len(x), 4)
    return np.concatenate((bias, temp), axis=1)   # test set

def y_trans(x, y):
    for j in range(len(x)):
        if x[j] != y:
            x[j] = 0
        else:
            x[j] = 1
    return np.array(x).reshape(len(x), 1)

test_set = add_bias(test)
train_set = add_bias(train)
train_target = y_trans(train_target_temp, 0)
test_target = y_trans(test_target_temp, 0)

weight_init = np.random.randn(5, 1)
z = np.dot(train_set, weight_init)

def sigmoid(z):                            # sigmoid 함수
    return 1 / (1 + np.exp(-z))

def cost_function(y, h):
    epsilon = 1e-32
    return np.sum(y*np.log(h + epsilon) + (1 - y)*np.log(1 - h + epsilon)) / (-1*y.shape[0])

def gradient_descent(x, y, weight, rate, epoch):                                  #learn method
    cost = []
    itera = []
    current_weight = weight
    for j in range(epoch):
        itera.append(j+1)
        z = np.dot(x, current_weight)
        a = sigmoid(z)
        cost.append(cost_function(y, a))
        temp = []
        for i in range(x.shape[1]):
            r = np.sum((a - y) * x[:, i].reshape(x.shape[0], 1))               # len(x)였는데 x.shape[0]으로 바꿈
            temp.append(r)
        sum1 = np.array(temp).reshape(len(temp), 1)
        current_weight = current_weight - (rate * sum1)
    plt.plot(itera, cost)
    plt.show()
    return current_weight


current = gradient_descent(train_set, train_target, weight_init, 0.00005, 1000)
q = np.dot(test_set, current)
print(sigmoid(q))
print(test_target)
#print(test_target)
#print(current)
#print(train_target.shape)
#print(test_target.shape)
#print(len(train_target_temp))