import numpy as np
from sklearn.datasets import load_iris

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

#
# def one_hot_encoding(y):  # multiclass로 학습 위해 output을 one hot encoding
#     temp = np.unique(y, axis=0)  # one_hot_encoding(target)
#     temp = temp.shape[0]
#     return np.eye(temp)[y]
#
# t = one_hot_encoding(train_target)
#
#
# # testset, testtarget = [], []
# size = 10
# #sample = np.random.randint(0, test_set.shape[0], size)                         # 100개의 sample index random으로 뽑음
# #print(sample)
# # for i in sample:
# #     testset.append(test_set[i])
# #     testtarget.append(t[i])
# # test_set_new = np.array(testset)
# # test_target_new = np.array(testtarget)
# #
# # print(test_set_new)
# # print(test_target_new)
#
# batch_size = min(size, train_set.shape[0])
# batch_mask = np.random.choice(train_set.shape[0], batch_size)
# x_batch = train_set[batch_mask]
# t_batch = t[batch_mask]
# print(x_batch)
# print(t_batch.shape)
print(test_set[0])


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)
[[-0.98857721 - 1.19498403  1.7798275]
 [1.92945432  2.08198445  0.48024869]]
x = np.array([0.6, 0.9])
p = net.predict(x)
np.argmax(p)
[1.14336256  1.15679559  1.50012032]
2
t = np.array([0, 0, 1])
net.loss(x, t)
0.87935693550588223


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
[[0.17430645  0.17666371 - 0.35097016]
 [0.26145968  0.26499557 - 0.52645524]]


출처: https: // sacko.tistory.com / 38[데이터
분석하는
문과생, 싸코]









