from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #추정치
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape)
print(x[:5])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

allAlgorithms = all_estimators(type_filter = 'classifier') # classifier

for (name, algorithm) in allAlgorithms:
    try : 
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ' ,accuracy_score(y_test, y_pred))
    except :
        print(name, '은 없음')

#Tensorflow 
#acc : 0.9912280440330505

'''
AdaBoostClassifier 의 정답률:  0.9473684210526315
BaggingClassifier 의 정답률:  0.9210526315789473
BernoulliNB 의 정답률:  0.6403508771929824
CalibratedClassifierCV 의 정답률:  0.8859649122807017
CategoricalNB 은 없음
CheckingClassifier 의 정답률:  0.35964912280701755
ClassifierChain 은 없음
ComplementNB 의 정답률:  0.868421052631579
DecisionTreeClassifier 의 정답률:  0.9473684210526315
DummyClassifier 의 정답률:  0.5
ExtraTreeClassifier 의 정답률:  0.956140350877193
ExtraTreesClassifier 의 정답률:  0.9649122807017544
GaussianNB 의 정답률:  0.9385964912280702
GaussianProcessClassifier 의 정답률:  0.8771929824561403
GradientBoostingClassifier 의 정답률:  0.956140350877193
HistGradientBoostingClassifier 의 정답률:  0.9736842105263158  <<<<<<<<<<< 최고치
KNeighborsClassifier 의 정답률:  0.9210526315789473
LabelPropagation 의 정답률:  0.3684210526315789
LabelSpreading 의 정답률:  0.3684210526315789
LinearDiscriminantAnalysis 의 정답률:  0.9473684210526315
LinearSVC 의 정답률:  0.9298245614035088
LogisticRegression 의 정답률:  0.9385964912280702
LogisticRegressionCV 의 정답률:  0.956140350877193
MLPClassifier 의 정답률:  0.9035087719298246
MultiOutputClassifier 은 없음
MultinomialNB 의 정답률:  0.8596491228070176
NearestCentroid 의 정답률:  0.868421052631579
NuSVC 의 정답률:  0.8596491228070176
OneVsOneClassifier 은 없음
OneVsRestClassifier 은 없음
OutputCodeClassifier 은 없음
PassiveAggressiveClassifier 의 정답률:  0.9122807017543859
Perceptron 의 정답률:  0.8947368421052632
QuadraticDiscriminantAnalysis 의 정답률:  0.9385964912280702
RadiusNeighborsClassifier 은 없음
RandomForestClassifier 의 정답률:  0.9736842105263158 <<<<<<<<<<< 최고치
RidgeClassifier 의 정답률:  0.956140350877193
RidgeClassifierCV 의 정답률:  0.9473684210526315
SGDClassifier 의 정답률:  0.8771929824561403
SVC 의 정답률:  0.8947368421052632
StackingClassifier 은 없음
VotingClassifier 은 없음
'''