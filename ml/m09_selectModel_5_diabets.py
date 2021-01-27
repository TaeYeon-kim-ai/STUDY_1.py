from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')

#1.데이터

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.2, random_state = 66)

allAlgorithms = all_estimators(type_filter = 'regressor')

for(name, algorithm) in allAlgorithms :
    try:
        model = algorithm()
            
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 없음')

#Tensorflow
# r2 : 0.5128401315682825

'''
ARDRegression 의 정답률 :  0.3740525977541973
AdaBoostRegressor 의 정답률 :  0.3884257661009941
BaggingRegressor 의 정답률 :  0.2712375950703747
BayesianRidge 의 정답률 :  0.3880392156414645
CCA 의 정답률 :  0.17913833844788818
DecisionTreeRegressor 의 정답률 :  0.07492580178446917
DummyRegressor 의 정답률 :  -0.00987357850732029
ElasticNet 의 정답률 :  -0.0007377358787463706
ElasticNetCV 의 정답률 :  0.43135542847414
ExtraTreeRegressor 의 정답률 :  -0.1420207781416989
ExtraTreesRegressor 의 정답률 :  0.3900225647337068
GammaRegressor 의 정답률 :  -0.0029018846458206404
GaussianProcessRegressor 의 정답률 :  -4.454301914011666
GeneralizedLinearRegressor 의 정답률 :  -0.0032222552995515574
GradientBoostingRegressor 의 정답률 :  0.21794207381475095
HistGradientBoostingRegressor 의 정답률 :  0.29401917994055804
HuberRegressor 의 정답률 :  0.3285185221568344
IsotonicRegression 은 없음
KNeighborsRegressor 의 정답률 :  0.35782491384655557
KernelRidge 의 정답률 :  -3.591164187374149
Lars 의 정답률 :  0.33352229072888784
LarsCV 의 정답률 :  0.3952314681939595
Lasso 의 정답률 :  0.3769258985715527
LassoCV 의 정답률 :  0.39541382838891836
LassoLars 의 정답률 :  0.4379497991524868
LassoLarsCV 의 정답률 :  0.3952314681939595
LassoLarsIC 의 정답률 :  0.38527866135765043
LinearRegression 의 정답률 :  0.3335222907288876
LinearSVR 의 정답률 :  -1.2124804680046277
MLPRegressor 의 정답률 :  -3.632283738324891
MultiOutputRegressor 은 없음
MultiTaskElasticNet 은 없음
MultiTaskElasticNetCV 은 없음
MultiTaskLasso 은 없음
MultiTaskLassoCV 은 없음
NuSVR 의 정답률 :  0.04853867233452491  <<<<<<<<<<<<<<<<<<<<<<<<<<<<최고치
OrthogonalMatchingPursuit 의 정답률 :  0.33638971626306335
OrthogonalMatchingPursuitCV 의 정답률 :  0.36528856250556374
PLSCanonical 의 정답률 :  -1.236554395581396
PLSRegression 의 정답률 :  0.37928738371535664
PassiveAggressiveRegressor 의 정답률 :  0.3534921982129331
PoissonRegressor 의 정답률 :  0.353562862111633
RANSACRegressor 의 정답률 :  -0.40352559633710094
RadiusNeighborsRegressor 의 정답률 :  -0.00987357850732029
RandomForestRegressor 의 정답률 :  0.34099815905068376
RegressorChain 은 없음
Ridge 의 정답률 :  0.27708239944728685
RidgeCV 의 정답률 :  0.44028332739174614
SGDRegressor 의 정답률 :  0.27412719893025006
SVR 의 정답률 :  0.060504085287063414
StackingRegressor 은 없음
TheilSenRegressor 의 정답률 :  0.27504211664641776
TransformedTargetRegressor 의 정답률 :  0.3335222907288876
TweedieRegressor 의 정답률 :  -0.0032222552995515574
VotingRegressor 은 없음
_SigmoidCalibration 은 없음
'''




