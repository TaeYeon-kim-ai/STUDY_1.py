from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
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
kfold = KFold(n_splits= 5, shuffle= True, random_state= 77)

allAlgorithms = all_estimators(type_filter = 'regressor') #classifier의 분류 모델 전체를 all_estomators에 넣는다.

for (name, algorithm) in allAlgorithms : #for 문에 넣는다 all_estimators의 name을 넣고 algorithm을 사용한다.
    try:  #try문 안에서 예외가 발생하면 except로 가서 처리하라
        model = algorithm() # for문에 의해 모든 모델을 돌려라
        score = cross_val_score(model, x_train, y_train, cv=kfold) #cv = 5로 넣어도 돌아가긴 하나, 옵션이 없음.
        #model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', score)
    except : 
        #continue #예외로된 모델명 안찍힘
        print(name, '은 없는 놈!') #except로 없는애들 찍고 다시 for문으로 돌아간다
'''
ARDRegression 의 정답률 :  [0.74662694 0.63754332 0.66286116 0.77005862 0.61576365]
AdaBoostRegressor 의 정답률 :  [0.86068444 0.80513979 0.83111    0.8206384  0.81065369]
BaggingRegressor 의 정답률 :  [0.88439008 0.84684181 0.85423777 0.86144634 0.81875342]
BayesianRidge 의 정답률 :  [0.71658716 0.62985707 0.71696677 0.78396776 0.62350229]
CCA 의 정답률 :  [0.75403154 0.60370745 0.59151525 0.78236827 0.58652477]
DecisionTreeRegressor 의 정답률 :  [0.77600783 0.85430491 0.63849078 0.72146993 0.86018238]
DummyRegressor 의 정답률 :  [-1.55205849e-02 -2.43226047e-06 -2.59834446e-02 -9.13739266e-04
 -2.34589093e-04]
ElasticNet 의 정답률 :  [0.6663248  0.60657827 0.69831825 0.72523377 0.61139593]
ElasticNetCV 의 정답률 :  [0.65514074 0.60044384 0.69084761 0.70803604 0.60162794]
ExtraTreeRegressor 의 정답률 :  [0.81407325 0.73549972 0.75317486 0.81490199 0.43481736]
ExtraTreesRegressor 의 정답률 :  [0.89461144 0.84349265 0.89698344 0.9122611  0.80130471]
GammaRegressor 의 정답률 :  [-1.65136734e-02 -2.49157998e-06 -2.37543752e-02 -9.81300631e-04
 -2.55467699e-04]
GaussianProcessRegressor 의 정답률 :  [-5.08066529 -6.33484961 -8.92429454 -5.37834544 -5.86897332]
GeneralizedLinearRegressor 의 정답률 :  [0.64759008 0.58408277 0.6855082  0.68449299 0.61426195]
GradientBoostingRegressor 의 정답률 :  [0.90127106 0.84619773 0.89990128 0.86444072 0.87540416]
HistGradientBoostingRegressor 의 정답률 :  [0.88454538 0.75772677 0.87713592 0.86276842 0.82856316]
HuberRegressor 의 정답률 :  [0.66334482 0.58858811 0.59470269 0.71510172 0.54776665]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.43657373 0.46585783 0.49681309 0.51124179 0.35861471]
KernelRidge 의 정답률 :  [0.70299474 0.5972021  0.65493216 0.77100787 0.59802237]
Lars 의 정답률 :  [0.74785581 0.66040768 0.67547128 0.77676921 0.62684136]
LarsCV 의 정답률 :  [0.71739278 0.6572376  0.68701485 0.77674681 0.6065536 ]
Lasso 의 정답률 :  [0.64800235 0.60573242 0.69177829 0.69882376 0.60581517]
LassoCV 의 정답률 :  [0.68015752 0.61539379 0.70518945 0.74152654 0.61933109]
LassoLars 의 정답률 :  [-1.55205849e-02 -2.43226047e-06 -2.59834446e-02 -9.13739266e-04
 -2.34589093e-04]
LassoLarsCV 의 정답률 :  [0.74566122 0.66040768 0.70295399 0.77671252 0.62863883]
LassoLarsIC 의 정답률 :  [0.74393361 0.65931048 0.68016808 0.77672388 0.61619103]
LinearRegression 의 정답률 :  [0.74785581 0.66040768 0.70254877 0.77676921 0.6270103 ]
LinearSVR 의 정답률 :  [ 0.04571519  0.34018818  0.49993656  0.67486424 -0.0590235 ]
MLPRegressor 의 정답률 :  [0.63625216 0.55922561 0.67854718 0.45310026 0.54646382]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.15139792 0.17046999 0.33906496 0.24070346 0.14202207]
OrthogonalMatchingPursuit 의 정답률 :  [0.55854059 0.48106068 0.53678613 0.54362952 0.50300862]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.69254896 0.55930871 0.61750308 0.78339249 0.58740262]
PLSCanonical 의 정답률 :  [-1.23408589 -2.66867359 -4.51618046 -1.30502955 -2.53460814]
PLSRegression 의 정답률 :  [0.69178866 0.6239101  0.64923557 0.79299719 0.56480161]
PassiveAggressiveRegressor 의 정답률 :  [-0.20384183  0.07052775  0.26229239  0.26357689 -0.05400547]
PoissonRegressor 의 정답률 :  [0.7670548  0.67353738 0.76117897 0.83301994 0.70332301]
RANSACRegressor 의 정답률 :  [0.5292652  0.53592013 0.36332544 0.80671902 0.4836671 ]
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor 의 정답률 :  [0.88860148 0.82776237 0.86677499 0.86741557 0.83095392]
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  [0.73471493 0.65022198 0.71567619 0.78341214 0.6254167 ]
RidgeCV 의 정답률 :  [0.74573605 0.65901402 0.70585334 0.77845331 0.62731105]
SGDRegressor 의 정답률 :  [-2.20105011e+24 -1.02929492e+27 -2.54500328e+27 -2.41484003e+26
 -3.67889323e+26]
SVR 의 정답률 :  [0.11585384 0.13322653 0.32796875 0.1983922  0.11640294]
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  [0.69456694 0.57791592 0.6388785  0.79833426 0.56700695]
TransformedTargetRegressor 의 정답률 :  [0.74785581 0.66040768 0.70254877 0.77676921 0.6270103 ]
TweedieRegressor 의 정답률 :  [0.64759008 0.58408277 0.6855082  0.68449299 0.61426195]
VotingRegressor 은 없는 놈!
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
'''
# Tensorflow
# acc :  1.0

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 