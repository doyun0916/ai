import numpy as np
from dataset.mnist import load_mnist
from KNN_Class_B411001 import KNN
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=False, normalize=True)
testset, testtarget = [], []
name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x_train_bi = np.array(x_train > 0, dtype=np.int8)


def crafted(dataset):
    result = []
    for i in range(len(dataset)):
        temp = []
        x1, x2, a, c = 0, 0, 0, 0
        b = 7
        for j in range(4):
            for k in range(4):
                for l in range(7):
                    x1 += np.count_nonzero(dataset[i][0][l+c][a:b])
                    x2 += np.count_nonzero(dataset[i][0][:, l + c][a:b])
                temp.append(x1)
                temp.append(x2)
                a += 7
                b += 7
                x1 = 0
                x2 = 0
            c += 7
            x1 = 0
            x2 = 0
            a = 0
            b = 7
        result.append(temp)
    return np.array(result)

x_train_crafted = crafted(x_train_bi)
x_test_crafted = crafted(x_test)

print(x_train_crafted[0])
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

