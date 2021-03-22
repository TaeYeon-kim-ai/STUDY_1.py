#실습 
#맹그러봐!!
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
wine_data = pd.read_csv("C:/data/data/winequality-white.csv", sep = ';')
x_test = pd.read_csv("C:/data/data/data-01-test-score.csv")
x = wine_data.iloc[:,0:-1]
y = wine_data.iloc[:,-1]

print(x)
print(y.shape)#(24, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size = 0.8, shuffle=True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(
        x,y, train_size = 0.8, shuffle=True, random_state = 66)


print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)#(3918, 11) (980, 11) (3918,) (980,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=11)
y_test = to_categorical(y_test, num_classes=11) # one-hot 인코딩
y_val = to_categorical(y_val, num_classes=11)

#MODEL
input1 = Input(shape=(x.shape[1] ,))
x = Dense(64, activation='relu')(input1)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.02)(x)
outputs = Dense(11, activation='relu')(x)
model = Model(inputs = input1, outputs = outputs)
        
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, verbose= 1, mode='auto')
model.fit(x_train, y_train, epochs = 100, batch_size = 128, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [es, lr])

# 평가
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# 단순 모델
# loss :  1.3697104454040527
# acc :  0.4969387650489807