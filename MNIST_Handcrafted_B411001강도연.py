import numpy as np
from dataset.mnist import load_mnist                                                                                    #load MNIST dataset
from KNN_Class_B411001 import KNN
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=False, normalize=True)

testset, testtarget = [], []                                                                                            # 100개의 test data sample을 담기 위한 array

name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']                                                               # output class 생성

x_train_bi = np.array(x_train > 0, dtype=np.int8)                                                                       # 해당박스에서 0이상인 pixel들의 합 계산하기 쉽도록 값을 0 아니면 1로 바꿔줌
x_test_bi = np.array(x_test > 0, dtype = np.int8)
def crafted_112(dataset):                                                                                               # 112개의 feature로 crafting
    result = []
    for i in range(len(dataset)):
        result_temp = []
        for j in range(28):
            a = 0
            b = 14
            for k in range(2):                                                                                          # 한 행을 두번에 걸쳐서 slicing한다. (0~13), (15~27)
                x1 = np.count_nonzero(dataset[i][0][j][a:b])
                result_temp.append(x1)
                a += b
                b += 14
        for j in range(28):
            a = 0
            b = 14
            for k in range(2):
                x2 = np.count_nonzero(dataset[i][0][:, j][a:b])                                                         # 한 열을 두번에 걸쳐서 slicing한다. (0~13), (15~27)
                result_temp.append(x2)
                a += b
                b += 14
        result.append(result_temp)
    return np.array(result)

#def crafted_56(dataset):                                                                                               # feature를 56개로 crafting하는 함수
#    result = []
#    for i in range(len(dataset)):
#        result_temp = []
#        for j in range(28):
#            x1 = np.count_nonzero(dataset[i][0][j])                                                                    # 각 행에 대해 iteration
#            result_temp.append(x1)
#        for j in range(28):
#            x2 = np.count_nonzero(dataset[i][0][:, j])                                                                 # 각 열에 대해 iteration
#            result_temp.append(x2)
#        result.append(result_temp)
#    return np.array(result)

#def crafted_32(dataset):                                                                                               # feature를 32개로 crafting하는 함수
#    result = []
#    for i in range(len(dataset)):
#        temp = []
#        x1, a, c = 0, 0, 0
#        b = 7
#        for j in range(4):
#            for k in range(4):
#                for l in range(7):                                                                                     #각 행 마다(7x7)4개의 box로 총 16개 생성과 0 이상의 pixel수들의 합 계산
#                    x1 += np.count_nonzero(dataset[i][0][l+c][a:b])
#                temp.append(x1)
#                a += 7
#                b += 7
#                x1 = 0
#            c += 7
#            a = 0
#            b = 7
#        c = 0
#        for j in range(4):
#            for k in range(4):
#                for l in range(7):                                                                                     #각 열 마다(7x7)4개의 box로 총 16개 생성과 0 이상의 pixel수들의 합 계산
#                    x1 += np.count_nonzero(dataset[i][0][:, l+c][a:b])
#                temp.append(x1)
#                a += 7
#                b += 7
#                x1 = 0
#            c += 7
#            a = 0
#            b = 7
#        result.append(temp)
#    return np.array(result)

#def crafted_28(dataset):                                                                                               # feature를 28개로 crafting하는 함수
#    x_data_result = []
#    for i in range(len(dataset)):
#        a = 0
#        x_train_temp = []
#        for j in range(14):
#            t1 = np.count_nonzero(dataset[i][a:13])
#            t2 = np.count_nonzero(dataset[i][a+28:a+41])
#            T1 = t1 + np.count_nonzero(t2)
#            x_train_temp.append(T1)
#            t3 = np.count_nonzero(dataset[i][a+14:a+27])
#            t4 = np.count_nonzero(dataset[i][a+42:a+55])
#            T2 = t3 + t4
#            x_train_temp.append(T2)
#            a += 56
#        x_data_result.append(x_train_temp)
#    return np.array(x_data_result)



x_train_crafted = crafted_112(x_train_bi)                                                                               # train dataset crafting
x_test_crafted = crafted_112(x_test_bi)                                                                                 # test dataset crafting

size = 100                                                                                                              # sample test 수는 100개
sample = np.random.randint(0, t_test.shape[0], size)                                                                    # 100개의 sample test data의 index들
for i in sample:
    testset.append(x_test_crafted[i])
    testtarget.append(t_test[i])
test_set = np.array(testset)
test_target = np.array(testtarget)

knn_MNIST = KNN(3, x_train_crafted, t_train, name)                 # KNN 클래스 MNIST 생성 (K = 3)
print()
print('Weighted Majority vote')
knn_MNIST.obtain_majority_vote(test_set, test_target, 'weighted')        # 각 test 데이터에 대한 결과값 및 실제값 비교 출력