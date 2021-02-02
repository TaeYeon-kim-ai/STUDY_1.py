import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
test = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)

x = train.drop(['id', 'digit', 'letter'], axis=1).values
z = test.drop(['id', 'letter'], axis = 1).values

#이미지 보기 왜안보이냐
plt.imshow(x[100].reshape(28,28))

x = x.reshape(-1, 28, 28, 1)
x = x/255
z = z.reshape(-1, 28, 28, 1)
z = z/255

#ImageDatagenrtator & data augmentation
# height_shift_range: 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동. 수치는 전체 높이의 비율(실수). 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동
# width_shift_range : 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동. 수치는 전체 높이의 비율(실수). 예를 들어 0.1이고 전체 높이가 100이면, 10픽셀 내외로 상하 이동
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()


'''




y = train['digit']

y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 0)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#(1638, 28, 28, 1) (410, 28, 28, 1) (1638,) (410,)

#tensorflow.keras .. to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


#모델링 CNN
inputs = Input(shape = (28,28,1))
conv2d = Conv2D(32, (4), strides =1 ,padding = 'SAME', input_shape = (28,28,1))(inputs)
conv2d = Conv2D(128, (3), strides =1, padding = 'SAME', activation='relu')(conv2d)
mp = MaxPooling2D(2)(conv2d)

conv2d = Conv2D(256, (3), strides =1, padding = 'SAME', activation='relu')(mp)
conv2d = Conv2D(256, (3), strides =1, padding = 'SAME', activation='relu')(conv2d)
conv2d = Conv2D(128, (3), strides =1, padding = 'SAME', activation='relu')(conv2d)
mp = MaxPooling2D(2)(conv2d)
drop = Dropout(0.3)(mp)

conv2d = Conv2D(64, (3), strides =1, padding = 'SAME', activation='relu')(mp)
conv2d = Conv2D(64, (3), strides =1, padding = 'SAME', activation='relu')(conv2d)
mp = MaxPooling2D(2)(conv2d)
drop = Dropout(0.3)(mp)
flt = Flatten()(drop)

dense = Dense(32, activation='relu')(flt)
dense = Dense(64, activation='relu')(dense)
dense = Dense(32, activation='relu')(dense)
dense = Dense(16, activation='relu')(dense)

outputs = Dense(10, activation='softmax')(dense)
model = Model(inputs = inputs, outputs = outputs)
model.summary()


#3. 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 64, validation_data=(x_val, y_val), verbose = 1 ,callbacks = [es, lr])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# y_pred = model.predict(x_test[:10])

cnn_test = test.drop(['id', 'letter'], axis=1).values
cnn_test = cnn_test.reshape(-1, 28, 28, 1)
cnn_test = cnn_test/255

submission = pd.read_csv('C:/STUDY/dacon/computer/submission.csv')
submission['digit'] = np.argmax(model.predict(cnn_test), axis=1)
submission.head()

submission.to_csv('C:/STUDY/dacon/computer/2021.02.02.csv', index=False)
'''