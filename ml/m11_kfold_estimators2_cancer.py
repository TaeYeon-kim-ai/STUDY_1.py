from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')
#warnings에 대해서 무시하겠다.

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
kfold = KFold(n_splits= 5, shuffle= True, random_state= 77)

allAlgorithms = all_estimators(type_filter = 'classifier') #classifier의 분류 모델 전체를 all_estomators에 넣는다.

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
AdaBoostClassifier 의 정답률 :  [0.97802198 0.95604396 0.95604396 0.91208791 0.95604396]
BaggingClassifier 의 정답률 :  [0.97802198 0.92307692 0.98901099 0.9010989  0.94505495]
BernoulliNB 의 정답률 :  [0.61538462 0.6043956  0.58241758 0.61538462 0.68131868]
CalibratedClassifierCV 의 정답률 :  [0.92307692 0.93406593 0.89010989 0.9010989  0.94505495]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  [0.87912088 0.9010989  0.84615385 0.86813187 0.91208791]
DecisionTreeClassifier 의 정답률 :  [0.86813187 0.91208791 0.95604396 0.87912088 0.92307692]
DummyClassifier 의 정답률 :  [0.50549451 0.51648352 0.50549451 0.48351648 0.51648352]
ExtraTreeClassifier 의 정답률 :  [0.87912088 0.93406593 0.94505495 0.87912088 0.83516484]
ExtraTreesClassifier 의 정답률 :  [0.98901099 0.96703297 0.97802198 0.93406593 0.93406593]
GaussianNB 의 정답률 :  [0.93406593 0.89010989 0.97802198 0.91208791 0.95604396]
GaussianProcessClassifier 의 정답률 :  [0.86813187 0.93406593 0.91208791 0.87912088 0.9010989 ]
GradientBoostingClassifier 의 정답률 :  [0.98901099 0.93406593 0.96703297 0.87912088 0.94505495]
HistGradientBoostingClassifier 의 정답률 :  [0.98901099 0.95604396 0.98901099 0.92307692 0.95604396]
KNeighborsClassifier 의 정답률 :  [0.89010989 0.95604396 0.91208791 0.89010989 0.92307692]
LabelPropagation 의 정답률 :  [0.3956044  0.41758242 0.42857143 0.3956044  0.32967033]
LabelSpreading 의 정답률 :  [0.3956044  0.41758242 0.42857143 0.3956044  0.32967033]
LinearDiscriminantAnalysis 의 정답률 :  [0.94505495 0.96703297 0.95604396 0.94505495 0.94505495]
LinearSVC 의 정답률 :  [0.83516484 0.93406593 0.84615385 0.84615385 0.93406593]
LogisticRegression 의 정답률 :  [0.9010989  0.96703297 0.93406593 0.94505495 0.92307692]
LogisticRegressionCV 의 정답률 :  [0.94505495 0.93406593 0.95604396 0.96703297 0.94505495]
MLPClassifier 의 정답률 :  [0.9010989  0.93406593 0.87912088 0.9010989  0.95604396]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  [0.87912088 0.9010989  0.84615385 0.85714286 0.91208791]
NearestCentroid 의 정답률 :  [0.86813187 0.91208791 0.89010989 0.84615385 0.91208791]
NuSVC 의 정답률 :  [0.84615385 0.87912088 0.89010989 0.84615385 0.9010989 ]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  [0.46153846 0.92307692 0.85714286 0.69230769 0.92307692]
Perceptron 의 정답률 :  [0.75824176 0.9010989  0.67032967 0.9010989  0.89010989]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.96703297 0.95604396 0.97802198 0.91208791 0.93406593]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 :  [0.98901099 0.92307692 0.97802198 0.92307692 0.96703297]
RidgeClassifier 의 정답률 :  [0.92307692 0.93406593 0.95604396 0.92307692 0.95604396]
RidgeClassifierCV 의 정답률 :  [0.94505495 0.93406593 0.96703297 0.94505495 0.96703297]
SGDClassifier 의 정답률 :  [0.78021978 0.87912088 0.85714286 0.9010989  0.9010989 ]
SVC 의 정답률 :  [0.93406593 0.94505495 0.9010989  0.84615385 0.92307692]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
# Tensorflow
# acc :  1.0

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 