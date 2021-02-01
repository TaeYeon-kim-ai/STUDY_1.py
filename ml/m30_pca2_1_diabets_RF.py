import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

# pca = PCA(n_components = 9, ) #n_components 압축 열 수 지정
# x2 = pca.fit_transform(x) #pca를 핏트랜스폼 (x)
# print(x2.shape) #(442, 7) #컬럼 재구성

# #압축된 것 중에서 어떤 피쳐가 중요한지
# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR) #컬럼 10개를 7개로 압축시켜서
# print(sum(pca_EVR))

#7개 # 0.9479436357350411 
#8개 # 0.9913119559917795
#9개 # 0.999143947009897

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #자세히 확인 explained_variance_rati
print("cumsum : ", cumsum) #max = 1
# cumcum :  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
# 0.94794364 0.99131196 0.99914395 1.        ] #cumsum작은것부터 하나씩 더해짐
# 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수

d = np.argmax(cumsum >= 0.95)+1 # 0.95중 최댓값 중 +1되는 걸로 써라(실질적인 가장 큰 값)
print("cumsum >=0.95", cumsum >=0.95)
print("d : ", d)
# cumsum >=0.95 [False False False False False False False  True  True  True]
# d :  8

#시각화
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

#헷갈림 주의
#feature importance : feature자체의 중요도를 찾는 것
#pca : 압축을 했을 때 몇개쓰면 좋은거냐 데이터 그대로쓰면 100%지만 줄일수록 손실이 있음. but 손실률 을 말하는것이지 score자체를 말하는건 아님 95%이상 기준?(개인)
#      - 원 데이터에서 데이터가 변형됨 / 전처리와 비슷한 개념.y 값과 매칭되는건 바뀌지 않음.

#데이터 - 전처리 - pca - 모델링 - 훈련 - 평가