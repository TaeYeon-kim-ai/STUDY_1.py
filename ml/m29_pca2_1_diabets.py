import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

#열 중요도 추출
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum : ", cumsum) #max = 1
# cumcum :  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
# 0.94794364 0.99131196 0.99914395 1.        ] #cumsum작은것부터 하나씩 더해짐
# 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >=0.95", cumsum >=0.95)
print("d : ", d)
# cumsum >=0.95 [False False False False False False False  True  True  True]
# d :  8

#시각화
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

pca = PCA(n_components = 9, ) #n_components 압축 열 수 지정
x2 = pca.fit_transform(x) #pca를 핏트랜스폼 (x)
print(x2.shape) #(442, 7) #컬럼 재구성

#압축된 것 중에서 어떤 피쳐가 중요한지
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) #컬럼 10개를 7개로 압축시켜서
print(sum(pca_EVR))

#7개 # 0.9479436357350411 
#8개 # 0.9913119559917795
#9개 # 0.999143947009897
#=========================================

#1. 데이터
print(x.shape)