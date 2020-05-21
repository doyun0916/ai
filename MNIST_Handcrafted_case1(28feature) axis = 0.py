import numpy as np
from dataset.mnist import load_mnist
from KNN_Class_B411001 import KNN
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True)
testset, testtarget = [], []
name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train_bi = np.array(x_train > 0, dtype=np.int8)
x_test_bi = np.array(x_test > 0, dtype=np.int8)
def crafting(dataset):
    x_data_result = []
    for i in range(len(dataset)):
        a = 0
        x_train_temp = []
        for j in range(14):
            t1 = np.count_nonzero(dataset[i][a:13])
            t2 = np.count_nonzero(dataset[i][a+28:a+41])
            T1 = t1 + np.count_nonzero(t2)
            x_train_temp.append(T1)
            t3 = np.count_nonzero(dataset[i][a+14:a+27])
            t4 = np.count_nonzero(dataset[i][a+42:a+55])
            T2 = t3 + t4
            x_train_temp.append(T2)
            a += 56
        x_data_result.append(x_train_temp)
    return np.array(x_data_result)

x_train_crafted = crafting(x_train_bi)
x_test_crafted = crafting(x_test_bi)

size = 100
sample = np.random.randint(0, t_test.shape[0], size)
for i in sample:
    testset.append(x_test_crafted[i])
    testtarget.append(t_test[i])
test_set = np.array(testset)
test_target = np.array(testtarget)
knn_MNIST = KNN(3, x_train_crafted, t_train, name)                 # KNN 클래스 MNIST 생성
print()
print('Weighted Majority vote')
knn_MNIST.obtain_majority_vote(test_set, test_target, 'weighted')        # 각 test 데이터에 대한 결과값 및 실제값 비교 출력