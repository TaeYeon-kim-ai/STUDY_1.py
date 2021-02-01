import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestClassifier

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

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