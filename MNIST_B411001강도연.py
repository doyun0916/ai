import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
from KNN_Class_B411001 import KNN
import sys, os
sys.path.append(os.pardir)
(x_train, t_train), (x_test, t_test) = \
 load_mnist(flatten=True, normalize=True)
testset, testtarget = [], []
name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
size = 10
sample = np.random.randint(0, t_test.shape[0], size)
for i in sample:
    testset.append(x_test[i])
    testtarget.append(t_test[i])
test_set = np.array(testset)
test_target = np.array(testtarget)
knn_MNIST = KNN(3, x_train, t_train, name)                 # KNN 클래스 knn_iris 생성
#print('Majority vote')
#knn_iris.obtain_majority_vote(test_set, test_target)      # Majority_vote를 이용하여
print()                                                   # 각 test 데이터에 대한 결과값 및 실제값 비교 출력
print('Weighted Majority vote')
knn_MNIST.obtain_majority_vote(test_set, test_target, 'weighted')      # 'weighted' parameter 추가시, Weighted방식으로 계산
                                                                      # 지정안해줄시, 기존 Majority vote 방식으로 계산


#image = x_train[0]
#def img_show(img):
 #pil_img = Image.fromarray(np.uint8(img))
 #pil_img.show()
#image = image.reshape(28, 28)
#img_show(image)