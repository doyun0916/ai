import numpy as np
from dataset.mnist import load_mnist                                    # MNIST data load
from Logistic_regression_class_B411001강도연 import Logistic_regression as lr
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True)

testset, testtarget = [], []                                            # 100개의 test data sample을 담기 위한 array

name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']               # output class 생성

LR_iris = lr()                                                          # Logistic regression class 선언

# learn(train_data, train_target, rate, epoch, 0(binary) or 1(multiclass), binary학습시 target class)
LR_iris.learn(x_train, t_train, 0.05, 100, 1)                           # multiclass로 학습
LR_iris.predict(x_test, t_test)                                         # '1'대신 '0' 과 target class 추가시 binary로 학습
