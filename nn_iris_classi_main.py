import numpy as np
from sklearn.datasets import load_iris
from nn_class_B411001강도연 import NeuralNetwork as Nn                # import Two_layered_NeuralNetwork_class
iris = load_iris()                                                   # iris data load
x = iris.data                                                        # iris data with features
y = iris.target                                                      # iris data's 실제 output
test, test_target, train, train_target = [], [], [], []              # test, test target = test 데이터셋, output 저장
                                                                     # train, train_target = train 데이터셋, output 저장
for i in range(x.shape[0]):                                          # test 데이터셋과 output, train 데이터셋과 output 분리
    if i % 5 == 4:                                                   # 총 150개 data중 매 5번째 data는 test data
        test.append([x[i]])
        test_target.append(y[i])
    else:
        train.append([x[i]])
        train_target.append(y[i])

train_set = np.array(train).reshape(len(train), len(train[0][0]))    # 분리된 train_set을 계산을 위해 2차원 array로 reshape
test_set = np.array(test).reshape(len(test), len(test[0][0]))        # 분리된 test_set을 계산을 위해 2차원 array로 reshape


def one_hot_encoding(y):                                        # learn method 내부에 one_hot_encoding을 넣어 놓아서,
    temp = np.unique(y, axis=0)                                 # 개별로 method를 test시,test_target을 one_hot_encoding
    temp = temp.shape[0]                                        # 한 후에, parameter로 전해주어야 합니다.
    return np.eye(temp)[y]                                  # 예시)  print(nn_iris.loss(test_set, test_target_encoded)),
                                                              # print(nn_iris.accuracy(test_set, test_target_encoded))


train_target_encoded = one_hot_encoding(train_target)
test_target_encoded = one_hot_encoding(test_target)

nn_iris = Nn(train_set.shape[1], 15, len(set(train_target)))  # NeuralNetwork init (input size,hidden size,output size)
nn_iris.learn(train_set, train_target, 0.0035, 7500, 100)     # learn(train_set, train_target, lr, epoch, mini-batch)
                                                              # 다른 함수 부를 필요 없이, learn 함수만으로 training하게 끔
                                                              # 더 효율적으로 작성해 보았습니다.
print("Training accuracy:", nn_iris.accuracy(train_set, train_target_encoded))  # 훈련된 weight, bias를 통한 train data accuracy 계산
print("Test accuracy:", nn_iris.accuracy(test_set, test_target_encoded))   # 훈련된 weight, bias를 통한 test data accuracy 계산
