import pandas as pd
import numpy as np
import os
import glob
import random

import warnings # 경고제어
warnings.filterwarnings("ignore")
train = pd.read_csv('./dacon/data/train/train.csv')
submission = pd.read_csv('./dacon/data/sample_submission.csv')
submission.tail()

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']] # 컬럼 추출

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # -48시간을 열1개 추가하여 채움 +1일 타겟 데이터
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') #  -48시간을 열2개 추가하여 채움  +2일 타겟 데이터
        temp = temp.dropna() #결측값 제거
        
        return temp.iloc[:-96] #예측날짜 제거 (-2일)

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']] 
                               
        return temp.iloc[-48:, :] # if문 진행 -48~끝까지, 뒤에서 6일째 데이터만 남김 - if 데이터 +1~2일 예측 ex_ 1일 - 2, 3일 예측 / 2일 - 3,4일 예측

df_train = preprocess_data(train)
df_train.iloc[:48] # T1, T2추가된 df_train 0~48까지 자르기
train.iloc[48:96] #기존 train 48~95까지 자르기
train.iloc[48+48:96+48] #train data 96 ~ +48까지 잘라서 보기

#============================================
#test데이터 병합
df_test = []

for i in range(81):
    file_path = './dacon/data/test/' + str(i) + '.csv' #test파일 불러오기
    temp = pd.read_csv(file_path) #pandas로 파일 읽기
    temp = preprocess_data(temp, is_train=False) 
    df_test.append(temp)

X_test = pd.concat(df_test)

#============================================

X_test.head(48) #x_test데이터
df_train.head()
df_train.iloc[-48:] 

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)1
#Target1, Target2데이터를 분리하여 X_train의 테스트값 분리 / Y_train의 테스트값 분리 +1일 / +2일에 대한  Tqrget train, test값

X_train_1.head()
X_test.head()

# ==========================================================
#quantile_loss적용
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test): #q값 
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2)) 
                #시리즈를 주어진 소수 자릿수로 반올림함 .00
    return pred, model
#LGBMTegressor LGBM회귀모델 적용 :  예측 변수가 주어졌을 때, 결과 변수의 q-분위수를 예측 (조건부 분위수 추정)

# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        #LGBM_models 마지막줄에 추가함
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)
        
    LGBM_actual_pred.columns = quantiles #quantiles추가 
    
    return LGBM_models, LGBM_actual_pred

#objective = 'quantile' quantile 숫자 적용 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48] # Target1 = 7일째 예측값, models_1과 results_1값에 기록

# Target2
models_2, results_2 = train_data(X_train2, Y_train2, X_valid2, Y_valid2, X_test)
results_2.sort_index()[:48] 

results_1.sort_index().iloc[:48]
results_2.sort_index()

print(results_1.shape, results_2.shape)
#(3888, 9) (3888, 9)


#submission에 작성
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
#contains: 다음문자가 포함된 곳에 데이터 작성, Day7의 q_0.1 : 부터 데이터 작성
#          submission의 id컬럼의 str로된 문자의 contains(행 길이만큼) 데이터 작성
#          .sort_index() 데이터를 results_1데이터를 정렬해서 넣기
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
#Day8의 q_0.1 : 부터 데이터 작성
submission
submission.iloc[:48]
submission.iloc[48:96]

submission.to_csv('./dacon/data/submission_v3.csv', index=False)