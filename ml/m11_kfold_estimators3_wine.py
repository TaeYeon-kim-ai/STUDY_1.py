from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')
#warnings에 대해서 무시하겠다.

#1. 데이터
dataset = load_wine()
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
AdaBoostClassifier 의 정답률 :  [0.82758621 0.96551724 0.96428571 0.75       0.39285714]
BaggingClassifier 의 정답률 :  [0.93103448 1.         0.96428571 0.92857143 0.92857143]
BernoulliNB 의 정답률 :  [0.37931034 0.4137931  0.32142857 0.42857143 0.42857143]
CalibratedClassifierCV 의 정답률 :  [0.93103448 0.89655172 0.96428571 0.89285714 0.89285714]
CategoricalNB 은 없는 놈!
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  [0.5862069  0.65517241 0.57142857 0.71428571 0.75      ]
DecisionTreeClassifier 의 정답률 :  [0.89655172 0.96551724 0.92857143 0.85714286 0.89285714]
DummyClassifier 의 정답률 :  [0.34482759 0.34482759 0.42857143 0.28571429 0.32142857]
ExtraTreeClassifier 의 정답률 :  [0.89655172 0.79310345 0.92857143 0.89285714 0.85714286]
ExtraTreesClassifier 의 정답률 :  [1.         1.         1.         0.96428571 1.        ]
GaussianNB 의 정답률 :  [1.         1.         1.         0.92857143 1.        ]
GaussianProcessClassifier 의 정답률 :  [0.51724138 0.48275862 0.46428571 0.5        0.32142857]
GradientBoostingClassifier 의 정답률 :  [0.89655172 1.         0.96428571 0.89285714 0.92857143]
HistGradientBoostingClassifier 의 정답률 :  [0.96551724 1.         1.         0.96428571 1.        ]
KNeighborsClassifier 의 정답률 :  [0.68965517 0.62068966 0.71428571 0.75       0.67857143]
LabelPropagation 의 정답률 :  [0.31034483 0.44827586 0.42857143 0.35714286 0.46428571]
LabelSpreading 의 정답률 :  [0.31034483 0.44827586 0.42857143 0.35714286 0.46428571]
LinearDiscriminantAnalysis 의 정답률 :  [1.         0.96551724 1.         0.96428571 0.96428571]
LinearSVC 의 정답률 :  [0.75862069 0.79310345 0.75       0.92857143 0.78571429]
LogisticRegression 의 정답률 :  [0.89655172 1.         1.         0.89285714 0.92857143]
LogisticRegressionCV 의 정답률 :  [0.93103448 1.         1.         0.89285714 0.92857143]
MLPClassifier 의 정답률 :  [0.34482759 0.93103448 0.32142857 0.92857143 0.39285714]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  [0.82758621 0.79310345 0.82142857 0.92857143 0.82142857]
NearestCentroid 의 정답률 :  [0.68965517 0.68965517 0.64285714 0.78571429 0.89285714]
NuSVC 의 정답률 :  [0.96551724 0.79310345 0.78571429 0.96428571 0.92857143]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  [0.44827586 0.24137931 0.53571429 0.64285714 0.25      ]
Perceptron 의 정답률 :  [0.5862069  0.68965517 0.53571429 0.46428571 0.71428571]
QuadraticDiscriminantAnalysis 의 정답률 :  [1.         1.         0.89285714 1.         1.        ]
RadiusNeighborsClassifier 은 없는 놈!
RandomForestClassifier 의 정답률 :  [1.         0.96551724 1.         0.96428571 1.        ]
RidgeClassifier 의 정답률 :  [1.         1.         1.         0.96428571 1.        ]
RidgeClassifierCV 의 정답률 :  [1.         1.         1.         0.96428571 1.        ]
SGDClassifier 의 정답률 :  [0.5862069  0.5862069  0.60714286 0.53571429 0.75      ]
SVC 의 정답률 :  [0.75862069 0.65517241 0.67857143 0.75       0.85714286]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''
#Tensorflow
#acc : 1.0

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 