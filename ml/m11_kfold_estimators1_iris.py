from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 둘중 하나 사용
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')
#warnings에 대해서 무시하겠다.

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
print(y.shape) #(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 44)
#iris train_test 나누기
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
AdaBoostClassifier 의 정답률 :  [0.91666667 0.91666667 0.875      1.         0.91666667]
BaggingClassifier 의 정답률 :  [0.91666667 0.91666667 0.875      1.         1.        ]
BernoulliNB 의 정답률 :  [0.16666667 0.29166667 0.29166667 0.25       0.33333333]
CalibratedClassifierCV 의 정답률 :  [0.95833333 0.83333333 0.75       1.         0.875     ]
CategoricalNB 의 정답률 :  [0.875      0.95833333 0.95833333 0.95833333 0.91666667]
CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB 의 정답률 :  [0.58333333 0.70833333 0.70833333 0.625      0.66666667]
DecisionTreeClassifier 의 정답률 :  [0.875      0.91666667 0.875      1.         1.        ]
DummyClassifier 의 정답률 :  [0.45833333 0.375      0.45833333 0.29166667 0.08333333]
ExtraTreeClassifier 의 정답률 :  [0.91666667 0.95833333 0.91666667 1.         1.        ]
ExtraTreesClassifier 의 정답률 :  [0.91666667 0.91666667 0.91666667 1.         1.        ]
GaussianNB 의 정답률 :  [1.         0.91666667 0.91666667 0.95833333 1.        ]
GaussianProcessClassifier 의 정답률 :  [0.91666667 0.95833333 0.95833333 1.         0.95833333]
GradientBoostingClassifier 의 정답률 :  [0.875      0.91666667 0.875      1.         1.        ]
HistGradientBoostingClassifier 의 정답률 :  [0.91666667 0.91666667 0.875      1.         1.        ]
KNeighborsClassifier 의 정답률 :  [0.875      0.91666667 0.95833333 1.         1.        ]
LabelPropagation 의 정답률 :  [0.91666667 0.91666667 0.95833333 1.         1.        ]
LabelSpreading 의 정답률 :  [0.91666667 0.91666667 0.95833333 1.         1.        ]
LinearDiscriminantAnalysis 의 정답률 :  [0.95833333 0.95833333 0.95833333 1.         1.        ]
LinearSVC 의 정답률 :  [0.95833333 0.91666667 0.875      1.         1.        ]
LogisticRegression 의 정답률 :  [0.91666667 0.91666667 0.91666667 1.         0.95833333]
LogisticRegressionCV 의 정답률 :  [0.875      0.91666667 0.91666667 1.         1.        ]
MLPClassifier 의 정답률 :  [0.91666667 0.95833333 1.         1.         1.        ]
MultiOutputClassifier 은 없는 놈!
MultinomialNB 의 정답률 :  [0.875      0.75       0.75       0.79166667 0.75      ]
NearestCentroid 의 정답률 :  [0.875      0.91666667 0.95833333 0.95833333 0.95833333]
NuSVC 의 정답률 :  [0.91666667 0.95833333 0.91666667 1.         0.91666667]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier 의 정답률 :  [0.91666667 0.75       0.79166667 0.625      0.70833333]
Perceptron 의 정답률 :  [0.91666667 0.875      0.875      0.625      0.875     ]
QuadraticDiscriminantAnalysis 의 정답률 :  [0.91666667 0.91666667 0.95833333 1.         0.95833333]
RadiusNeighborsClassifier 의 정답률 :  [0.95833333 0.91666667 0.91666667 1.         0.95833333]
RandomForestClassifier 의 정답률 :  [0.91666667 0.91666667 0.875      1.         1.        ]
RidgeClassifier 의 정답률 :  [0.95833333 0.75       0.79166667 0.875      0.875     ]
RidgeClassifierCV 의 정답률 :  [0.95833333 0.75       0.79166667 0.875      0.875     ]
SGDClassifier 의 정답률 :  [0.95833333 0.66666667 0.75       1.         0.75      ]
SVC 의 정답률 :  [0.91666667 0.95833333 0.91666667 1.         1.        ]
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