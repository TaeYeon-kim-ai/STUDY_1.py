from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')
#warnings에 대해서 무시하겠다.

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
#iris train_test 나누기

allAlgorithms = all_estimators(type_filter = 'regressor') #classifier의 분류 모델 전체를 all_estomators에 넣는다.

for (name, algorithm) in allAlgorithms : #for 문에 넣는다 all_estimators의 name을 넣고 algorithm을 사용한다.
    try:  #try문 안에서 예외가 발생하면 except로 가서 처리하라
        model = algorithm() # for문에 의해 모든 모델을 돌려라

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except : 
        #continue #예외로된 모델명 안찍힘
        print(name, '은 없는 놈!') #except로 없는애들 찍고 다시 for문으로 돌아간다

#Tensorflow 
# R2 : 0.9430991642272919

'''
ARDRegression 의 정답률 :  0.7512651671068512
AdaBoostRegressor 의 정답률 :  0.8377449917834314
BaggingRegressor 의 정답률 :  0.8723559082948449
BayesianRidge 의 정답률 :  0.7444785336818134
CCA 의 정답률 :  0.7270542664211515
DecisionTreeRegressor 의 정답률 :  0.8232063431533674
DummyRegressor 의 정답률 :  -0.0007982049217318821
ElasticNet 의 정답률 :  0.6990500898755508
ElasticNetCV 의 정답률 :  0.6902681369495265
ExtraTreeRegressor 의 정답률 :  0.7230809176815004
ExtraTreesRegressor 의 정답률 :  0.9004218826324184
GammaRegressor 의 정답률 :  -0.0007982049217318821
GaussianProcessRegressor 의 정답률 :  -5.639147690233129
GeneralizedLinearRegressor 의 정답률 :  0.6917882790641979
GradientBoostingRegressor 의 정답률 :  0.8949971651294957
HistGradientBoostingRegressor 의 정답률 :  0.8991491407747458
HuberRegressor 의 정답률 :  0.737051252959666
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 :  0.6390759816821279
KernelRidge 의 정답률 :  0.7744886786248365
Lars 의 정답률 :  0.7521800808693162
LarsCV 의 정답률 :  0.7570138649983489
Lasso 의 정답률 :  0.6855879495660049
LassoCV 의 정답률 :  0.7154057460487298
LassoLars 의 정답률 :  -0.0007982049217318821
LassoLarsCV 의 정답률 :  0.7570138649983489
LassoLarsIC 의 정답률 :  0.7540945959884463
LinearRegression 의 정답률 :  0.7521800808693132
LinearSVR 의 정답률 :  -1.1514254598301878
MLPRegressor 의 정답률 :  0.3943017427939759
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 :  0.32534704254368274
OrthogonalMatchingPursuit 의 정답률 :  0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률 :  0.7377665753906506
PLSCanonical 의 정답률 :  -1.7155095545127725
PLSRegression 의 정답률 :  0.766694031040294
PassiveAggressiveRegressor 의 정답률 :  -0.20521032993388055
PoissonRegressor 의 정답률 :  0.8014714721143593
RANSACRegressor 의 정답률 :  0.7530801429139966
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 :  0.889678798796214
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  0.7539303499010777
RidgeCV 의 정답률 :  0.7530092299355243
SGDRegressor 의 정답률 :  -5.012188055157342e+26
SVR 의 정답률 :  0.2868662719877668
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  0.7969690462705734
TransformedTargetRegressor 의 정답률 :  0.7521800808693132
TweedieRegressor 의 정답률 :  0.6917882790641979
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 