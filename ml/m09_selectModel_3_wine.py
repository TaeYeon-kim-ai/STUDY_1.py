from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms :
    try : 
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except : 
        print(name, '은 없음')


#Tensorflow
#acc : 1.0

'''
AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  1.0               <<<<<<<<<< 최고치
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB 은 없음
CheckingClassifier 의 정답률 :  0.3888888888888889
ClassifierChain 은 없음
ComplementNB 의 정답률 :  0.6944444444444444
DecisionTreeClassifier 의 정답률 :  0.9722222222222222
DummyClassifier 의 정답률 :  0.3611111111111111
ExtraTreeClassifier 의 정답률 :  0.9166666666666666
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.4444444444444444
GradientBoostingClassifier 의 정답률 :  0.9722222222222222
HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
KNeighborsClassifier 의 정답률 :  0.6944444444444444
LabelPropagation 의 정답률 :  0.5277777777777778
LabelSpreading 의 정답률 :  0.5277777777777778
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.8888888888888888
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.8333333333333334
MultiOutputClassifier 은 없음
MultinomialNB 의 정답률 :  0.7777777777777778
NearestCentroid 의 정답률 :  0.6944444444444444
NuSVC 의 정답률 :  0.9444444444444444
OneVsOneClassifier 은 없음
OneVsRestClassifier 은 없음
OutputCodeClassifier 은 없음
PassiveAggressiveClassifier 의 정답률 :  0.6944444444444444
Perceptron 의 정답률 :  0.6388888888888888
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier 은 없음
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  1.0
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  0.6388888888888888
SVC 의 정답률 :  0.6944444444444444
StackingClassifier 은 없음
VotingClassifier 은 없음
'''