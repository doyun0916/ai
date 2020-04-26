import math
from statistics import mode                                                                                             # Majority를 뽑아내기 위한 함수  ex) mode(1, 0, 1, 1) = 1
import numpy as np

class KNN():
    def __init__(self, k, train_set, train_target, name):                                                               # KNN class input으로
        self.k = k                                                                                                      # K
        self.train_set = train_set                                                                                      # train데이터셋과
        self.train_target = train_target                                                                                # train데이터셋에 대한 output,
        self.name = name                                                                                                # 3개 output class들의 이름 (꽃 이름)을 받아서 저장

    def obtain_K_nearest_neighbor(self, test):                                                                          # test데이터셋을 input으로 받는다
        result = []
        for i in range(test.shape[0]):                                                                                  # 각 10개(test.shape[0])의 test데이터에 대해
            dist_sort = sorted(self.cal_dis(i, test, self.train_set, self.train_target))                                # 140개의 train데이터들과의 거리를 구하고(cal_dis), ascending order로 sort하여
            p_sli = dist_sort[:self.k]                                                                                  # k개만큼 빼낸다.
            result.append(p_sli)
        return result                                                                                                   # result는 예를 들어 K=3이면, 3묶음씩 10개(test데이터마다의)의 data를 가지고 있다.
                                                                                                                        # ex) [[1.002, 0], [4.332, 2], ....]
    def cal_dis(self, i, test, train, train_target):                                                                    # test데이터 1개에 대해 140개의 train데이터와의 거리를 구하는 method
        dist, dist_var = [], []
        for j in range(train.shape[0]):                                                                                 # 140번(train.shape[0])만큼 반복하되,
            dist_var = math.sqrt(np.sum((test[i] - train[j])**2))                                                       # feature 4개 가지고 Euclidean 거리 계산하고,
            dist.append([dist_var, train_target[j]])                                                                    # train_target[j](해당 train data가 속해있는 class)를 나중의 편의를 위해 같이 저장
        return dist                                                                                                     # ex) [1.002, 0] == [distance, class]
                                                                                                                        # dist = 140개의 거리들과 해당 train_target값 가진 list
    def obtain_majority_vote(self, test, test_target, w=None):                                                          # obtain_K_nearest_neighbor함수를 이용하여 얻은 결과를 사용하여
        opt = 0                                                                                                         # 계산된 output과 실제 output을 출력하여 비교하도록 보여주는 method
        sum = []
        temp = 0
        if w == 'weighted':                                                                                             # 마지막 w parameter에 'weighted'가 들어오면 opt = 1로 ,else opt = 0
            opt = 1
        target_count = []                                                                                               # target_count = 가장 많이 속해있는 class를 찾기위해 class들만 모아놓기위한 list
        result = self.obtain_K_nearest_neighbor(test)
        if opt == 1:                                                                                                    # weighed 방식일때,
            for b in range(len(result)):                                                                                # result(거리값)들을 가중치로 바꿔놓는다.
                for d in range(self.k):
                    result[b][d][0] = 1/(result[b][d][0] + 1)                                                           # 가중치는 1/(d+1)을 이용하였다.
            for z in range(len(result)):                                                                                # 바꾼 result값에서, 각 test data set에서의 class별로 가중치의 합을 더하여 저장한다.
                for s in range(len(self.name)):
                    for t in range(self.k):
                        if result[z][t][1] == s:
                            temp += result[z][t][0]
                    sum.append(temp)                                                                                    #sum = testdata별로 class각각의 가중치의 합이 저장된다.
                    temp = 0
                print('Test Data Index:', z, '   Computed class:', self.name[sum.index(max(sum))],                      #sum.index(max(sum))을 이용하여, 어떠한 class가 가중치가 가장 큰지 찾아낸다.
                      '     True class:', self.name[test_target[z]])
                sum = []
        else:
            for a in range(len(result)):                                                                                # 10번동안
                for c in range(self.k):                                                                                 # K개 만큼 class들을 뽑아낸다.
                    target_count.append(result[a][c][1])
                print('Test Data Index:', a, '   Computed class:', self.name[mode(target_count)],                       # mode함수를 이용하여 class들중 많은 수의 class를 뽑아낸다.(Majority vote)
                      '     True class:', self.name[test_target[a]])
                target_count = []