from sklearn.model_selection import train_test_split
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

allAlgorithms = all_estimators(type_filter = 'classifier') #classifier의 분류 모델 전체를 all_estomators에 넣는다.

for (name, algorithm) in allAlgorithms : #for 문에 넣는다 all_estimators의 name을 넣고 algorithm을 사용한다.
    try:  #try문 안에서 예외가 발생하면 except로 가서 처리하라
        model = algorithm() # for문에 의해 모든 모델을 돌려라

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except : 
        #continue #예외로된 모델명 안찍힘
        print(name, '은 없는 놈!') #except로 없는애들 찍고 다시 for문으로 돌아간다

# Tensorflow
# acc :  1.0
'''
# AdaBoostClassifier 의 정답률 :  0.9666666666666667
# BaggingClassifier 의 정답률 :  0.9666666666666667
# BernoulliNB 의 정답률 :  0.3
# CalibratedClassifierCV 의 정답률 :  0.9333333333333333
# CategoricalNB 의 정답률 :  0.9
# CheckingClassifier 의 정답률 :  0.3
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :  0.7
# DecisionTreeClassifier 의 정답률 :  0.8666666666666667
# DummyClassifier 의 정답률 :  0.2
# ExtraTreeClassifier 의 정답률 :  0.9666666666666667
# ExtraTreesClassifier 의 정답률 :  0.9666666666666667
# GaussianNB 의 정답률 :  0.9333333333333333
# GaussianProcessClassifier 의 정답률 :  0.9666666666666667
# GradientBoostingClassifier 의 정답률 :  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
# KNeighborsClassifier 의 정답률 :  0.9666666666666667
# LabelPropagation 의 정답률 :  0.9666666666666667
# LabelSpreading 의 정답률 :  0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 :  1.0 0.9736842105263158  <<<<<<<<<<< 최고치
# LinearSVC 의 정답률 :  0.9666666666666667
# LogisticRegression 의 정답률 :  0.9666666666666667
# LogisticRegressionCV 의 정답률 :  0.9666666666666667
# MLPClassifier 의 정답률 :  1.0
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :  0.8666666666666667
# NearestCentroid 의 정답률 :  0.9
# NuSVC 의 정답률 :  0.9666666666666667
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :  0.9333333333333333
# Perceptron 의 정답률 :  0.7333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 :  1.0
# RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
# RandomForestClassifier 의 정답률 :  0.9666666666666667
# RidgeClassifier 의 정답률 :  0.8333333333333334
# RidgeClassifierCV 의 정답률 :  0.8333333333333334
# SGDClassifier 의 정답률 :  0.9666666666666667
# SVC 의 정답률 :  0.9666666666666667
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!
'''

#"7번째 부터 안돌아간다??"
#TypeError: __init__() missing 1 required positional argument: 'base_estimator'

import sklearn
print(sklearn.__version__) #0.23.1 #all_estimators는 0.20정도에 모든 것들이 먹힌다.
#해결법 (?) 