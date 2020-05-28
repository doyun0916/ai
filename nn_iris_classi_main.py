import numpy as np
from sklearn.datasets import load_iris
from nn_class_B411001강도연 import NeuralNetwork as Nn      # import logistic_regression_class
iris = load_iris()                                                   # iris data load
x = iris.data                                                        # iris data with features
y = iris.target                                                      # iris data's 실제 output
iris_name = iris.target_names                                        # iris 3종류의 이름
test, test_target, train, train_target = [], [], [], []              # test, test target = test 데이터셋, output 저장
for i in range(x.shape[0]):                                          # train, train_target = train 데이터셋, output 저장
    if i % 5 == 4:                                                # test 데이터셋과 output, train 데이터셋과 output 분리
        test.append([x[i]])
        test_target.append(y[i])
    else:
        train.append([x[i]])
        train_target.append(y[i])

train_set = np.array(train).reshape(len(train), len(train[0][0]))    # 분리된 train_set을 계산을 위해 알맞은 형태로 reshape
test_set = np.array(test).reshape(len(test), len(test[0][0]))        # 분리된 test_set을 계산을 위해 알맞은 형태로 reshape


def one_hot_encoding(y):  # multiclass로 학습 위해 output을 one hot encoding
    temp = np.unique(y, axis=0)  # one_hot_encoding(target)
    temp = temp.shape[0]
    return np.eye(temp)[y]

t = one_hot_encoding(test_target)

nn_iris = Nn(train_set.shape[1], 10, len(set(train_target)))
#print(nn_iris.loss(test_set, t))
#print(nn_iris.predict(test_set))
nn_iris.learn(train_set, train_target, 0.005, 10000, 100)
print(nn_iris.accuracy(test_set, t))
