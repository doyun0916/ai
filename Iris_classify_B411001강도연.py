import numpy as np
from sklearn.datasets import load_iris
from KNN_Class_B411001 import KNN
iris = load_iris()                                                   # iris data load
x = iris.data                                                        # iris data with features
y = iris.target                                                      # iris data's 실제 output
iris_name = iris.target_names                                        # iris 3종류의 이름
test, test_target, train, train_target = [], [], [], []              # test, test target = test 데이터셋, output 저장
for i in range(x.shape[0]):                                          # train, train_target = train 데이터셋, output 저장
    if i % 15 == 14:                 # test 데이터셋과 output, train 데이터셋과 output 분리
        test.append([x[i]])
        test_target.append(y[i])
    else:
        train.append([x[i]])
        train_target.append(y[i])
test_set = np.array(test)                        # test set
train_set = np.array(train)                      # train set


knn_iris = KNN(5, train_set, train_target, iris_name)                 # KNN 클래스 knn_iris 생성
print('Majority vote')
knn_iris.obtain_majority_vote(test_set, test_target)      # Majority_vote를 이용하여
print()                                                   # 각 test 데이터에 대한 결과값 및 실제값 비교 출력
print('Weighted Majority vote')
knn_iris.obtain_majority_vote(test_set, test_target, 'weighted')      # 'weighted' parameter 추가시, Weighted방식으로 계산
                                                                      # 지정안해줄시, 기존 Majority vote 방식으로 계산
