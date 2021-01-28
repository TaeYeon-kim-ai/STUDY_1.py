from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')
#warnings에 대해서 무시하겠다.

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
kfold = KFold(n_splits= 5, shuffle= True, random_state= 77)

allAlgorithms = all_estimators(type_filter = 'regressor') #classifier의 분류 모델 전체를 all_estomators에 넣는다.

for (name, algorithm) in allAlgorithms : #for 문에 넣는다 all_estimators의 name을 넣고 algorithm을 사용한다.
    try:  #try문 안에서 예외가 발생하면 except로 가서 처리하라
        model = algorithm() # for문에 의해 모든 모델을 돌려라
        score = cross_val_score(model, x_train, y_train, cv=kfold) #cv = 5로 넣어도 돌아가긴 하나, 옵션이 없음.
        # y_pred = model.predict(x_test)
        # r2 = r2_score(y_test, y_pred)
        # print("r2 : ", r2)
        print(name, '의 정답률 : ', score)
    except : 
        #continue #예외로된 모델명 안찍힘
        print(name, '은 없음') #except로 없는애들 찍고 다시 for문으로 돌아간다

'''
ARDRegression 의 정답률 :  [0.47481641 0.48527985 0.46792353 0.5600166  0.37031528]
AdaBoostRegressor 의 정답률 :  [0.48160747 0.44128304 0.45815722 0.51604346 0.19939256]
BaggingRegressor 의 정답률 :  [0.39667805 0.31792801 0.40407017 0.56838454 0.24050214]
BayesianRidge 의 정답률 :  [0.49043067 0.47221963 0.47734842 0.56352615 0.37761203]
CCA 의 정답률 :  [0.37115606 0.17980581 0.46471171 0.55258411 0.29418323]
DecisionTreeRegressor 의 정답률 :  [-0.13892744  0.0624486   0.15354914 -0.13637076 -0.54019279]
DummyRegressor 의 정답률 :  [-0.00432459 -0.01078332 -0.00293931 -0.03545274 -0.00601344]
ElasticNet 의 정답률 :  [ 0.00412805 -0.00069092  0.00541169 -0.02534326  0.00180807]
ElasticNetCV 의 정답률 :  [0.42321988 0.45567821 0.44331144 0.49134338 0.31966629]
ExtraTreeRegressor 의 정답률 :  [ 0.03972972 -0.28687271 -0.12092226 -0.12652664 -0.67218141]
ExtraTreesRegressor 의 정답률 :  [0.49229038 0.44395952 0.46672765 0.48343326 0.21669252]
GammaRegressor 의 정답률 :  [ 0.00162114 -0.00264572  0.00327988 -0.02954757 -0.00031393]
GaussianProcessRegressor 의 정답률 :  [-11.69741011 -27.41978887 -11.04353971 -11.65393816 -13.29125201]
GeneralizedLinearRegressor 의 정답률 :  [ 0.00179076 -0.00309452  0.00342069 -0.02803447 -0.00051296]
GradientBoostingRegressor 의 정답률 :  [0.50992281 0.40506295 0.45600372 0.55666955 0.24902229]
HistGradientBoostingRegressor 의 정답률 :  [0.41139154 0.41043048 0.40943158 0.47123317 0.23773978]
HuberRegressor 의 정답률 :  [0.49876864 0.45750882 0.45846722 0.5446248  0.40434557]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.35817382 0.33395457 0.38080499 0.37659325 0.15623429]
KernelRidge 의 정답률 :  [-3.21024896 -3.52623286 -2.96877666 -4.1852857  -3.538776  ]
Lars 의 정답률 :  [ 0.24103098  0.47715173 -2.88338963  0.53499403  0.38655217]
LarsCV 의 정답률 :  [0.45258889 0.47769925 0.47726076 0.55002657 0.370046  ]
Lasso 의 정답률 :  [0.34093535 0.35048334 0.32720416 0.36673092 0.30181413]
LassoCV 의 정답률 :  [0.457804   0.47781607 0.47313797 0.56166389 0.36822158]
LassoLars 의 정답률 :  [0.38053894 0.39305551 0.36927422 0.42082682 0.32096914]
LassoLarsCV 의 정답률 :  [0.45258889 0.47715146 0.47338409 0.56171182 0.36757517]
LassoLarsIC 의 정답률 :  [0.44194233 0.47724265 0.47868004 0.55061817 0.36793503]
LinearRegression 의 정답률 :  [0.49652963 0.47715173 0.46565514 0.55184515 0.38655217]
LinearSVR 의 정답률 :  [-0.37544598 -0.35253931 -0.40858905 -0.7523567  -0.37216917]
MLPRegressor 의 정답률 :  [-2.77163459 -2.88865225 -2.6465854  -3.52224762 -2.70306923]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.1379982  0.14066337 0.12730405 0.10450657 0.10704505]
OrthogonalMatchingPursuit 의 정답률 :  [0.33716192 0.3056527  0.33951272 0.31041136 0.20335767]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.44583958 0.44827293 0.46503415 0.55425066 0.37465953]
PLSCanonical 의 정답률 :  [-0.99728739 -1.99889607 -0.68952862 -1.1601449  -1.86824395]
PLSRegression 의 정답률 :  [0.48555824 0.44765396 0.47619168 0.58269817 0.37019117]
PassiveAggressiveRegressor 의 정답률 :  [0.46154548 0.46925769 0.4635812  0.49479588 0.27409339]
PoissonRegressor 의 정답률 :  [0.30752758 0.35929857 0.34386618 0.37679995 0.2465751 ]
RANSACRegressor 의 정답률 :  [0.19680092 0.19811341 0.25148156 0.3060024  0.05355343]
RadiusNeighborsRegressor 의 정답률 :  [-0.00432459 -0.01078332 -0.00293931 -0.03545274 -0.00601344]
RandomForestRegressor 의 정답률 :  [0.46940643 0.44404027 0.41755082 0.54838369 0.25183083]
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  [0.38268858 0.42942638 0.40494235 0.4421777  0.29404632]
RidgeCV 의 정답률 :  [0.48578296 0.47510598 0.48087375 0.55916314 0.37137981]
SGDRegressor 의 정답률 :  [0.36747827 0.42533638 0.40301517 0.4295514  0.27220149]
SVR 의 정답률 :  [0.13146673 0.15492254 0.09948322 0.02698625 0.11497241]
'''
#Tensorflow
# r2 : 0.5128401315682825

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 

#4. 평가
y_pred = model.predict(x_test)
#print(y_pred)
# print(y)

# loss, acc = model.evaluate(x_test, y_test) #본래 loss 와 acc이지만
result = model.score(x_test, y_test) #자동으로 evaluate 해서 acc 빼준다.
print('result : ', result)

#accuracy_score
#                  (실데이터, 예측결과 데이터)

