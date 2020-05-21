import numpy as np
from sklearn.datasets import load_iris
from Logistic_regression_class_B411001강도연 import Logistic_regression as lr      # import logistic_regression_class
iris = load_iris()                                                   # iris data load
x = iris.data                                                        # iris data with features
y = iris.target                                                      # iris data's 실제 output
iris_name = iris.target_names                                        # iris 3종류의 이름
test, test_target, train, train_target = [], [], [], []              # test, test target = test 데이터셋, output 저장
for i in range(x.shape[0]):                                          # train, train_target = train 데이터셋, output 저장
    if i % 15 == 14:                                                # test 데이터셋과 output, train 데이터셋과 output 분리
        test.append([x[i]])
        test_target.append(y[i])
    else:
        train.append([x[i]])
        train_target.append(y[i])

train_set = np.array(train).reshape(len(train), len(train[0][0]))    # 분리된 train_set을 계산을 위해 알맞은 형태로 reshape
test_set = np.array(test).reshape(len(test), len(test[0][0]))        # 분리된 test_set을 계산을 위해 알맞은 형태로 reshape

LR_iris = lr()                                                       # Logistic regression class 선언

# learn(train_data, train_target, rate, epoch, 0(binary) or 1(multiclass), binary학습시 target class)
LR_iris.learn(train_set, train_target, 0.00058, 1000, 1)             # multiclass로 학습
LR_iris.predict(test_set, test_target)                               # '1'대신 '0' 과 target class 추가시 binary로 학습