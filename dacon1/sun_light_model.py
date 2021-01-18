import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

#1.데이터
data1 = np.load('./dacon1/data/test/sun_train_data.npy')
data2 = np.load('./dacon1/data/test/sun_test_data.npy')

#1.1 함수정의


#1.2 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, randam_state = 100)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, randam_state = 100)