import numpy as np
from dataset.mnist import load_mnist                      # MNIST data load
from KNN_Class_B411001 import KNN
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True)

testset, testtarget = [], []                                                 # 100개의 test data sample을 담기 위한 array

name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']                    # output class 생성

size = 100
sample = np.random.randint(0, t_test.shape[0], size)                         # 100개의 sample index random으로 뽑음
for i in sample:
    testset.append(x_test[i])
    testtarget.append(t_test[i])
test_set = np.array(testset)
test_target = np.array(testtarget)

#knn_MNIST = KNN(3, x_train, t_train, name)                      # KNN 클래스 MNIST 생성(K = 3)
#print()
#print('Weighted Majority vote')
#knn_MNIST.obtain_majority_vote(test_set, test_target, 'weighted')       # 각 test 데이터에 대한 결과값 및 실제값 비교 출력
print(test_set[0])