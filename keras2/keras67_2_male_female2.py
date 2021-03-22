#실습
#남자 여자 구분
#ImageDataGenerator의 fit사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#http://naver.me/5B1Y91UT
#이미지 / 데이터로 전처리 / 증폭가능하게 //

#train/test준비 선언
# train_datagen = ImageDataGenerator(
#     rescale= 1./255, #전처리 
#     # horizontal_flip=True,
#     # vertical_flip=True,
#     width_shift_range=[-1, 1],
#     height_shift_range=[-1, 1],
#     rotation_range=3,
#     zoom_range=0.3,
#     # shear_range=0.7,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(rescale=1./255) 

# xy_train = train_datagen.flow_from_directory( 
#     'C:/data/image/gender_generator',
#     target_size = (64, 64),
#     batch_size = 2500,
#     class_mode= 'binary' ,
# )

# xy_test = test_datagen.flow_from_directory(
#     'C:/data/image/gender',
#     target_size = (64, 64),
#     batch_size = 2000,
#     class_mode= 'binary' ,
# )

# print(train_datagen) #Found 5208 images belonging to 2 classes.
# print(test_datagen) #Found 1736 images belonging to 2 classes.

# # print(xy_train)
# #npy 저장
# np.save('C:/data/image/gender_npy/keras67_train_x.npy', arr=xy_train[0][0]) #x 1
# np.save('C:/data/image/gender_npy/keras67_train_y.npy', arr=xy_train[0][1])
# np.save('C:/data/image/gender_npy/keras67_test_x.npy', arr=xy_test[0][0]) #x 1
# np.save('C:/data/image/gender_npy/keras67_test_y.npy', arr=xy_test[0][1])

#npy 로드
x_train = np.load('C:/data/image/gender_npy/keras67_train_x.npy') #x 1
y_train = np.load('C:/data/image/gender_npy/keras67_train_y.npy')
x_test = np.load('C:/data/image/gender_npy/keras67_test_x.npy') #x 1
y_test = np.load('C:/data/image/gender_npy/keras67_test_y.npy')

print(x_train.shape, y_train.shape) #(1736, 64, 64, 3) (1736,)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False, random_state = 66)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
input1 = Input(shape=(x_train.shape[1], x_train.shape[2] ,x_train.shape[3]))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(128, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 3, activation='relu')(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.2)(x)

x = Conv2D(64, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, activation='relu')(x)
x = Dropout(0.2)(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)

x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x) 
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'acc', patience = 50, mode = 'auto')
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=20, verbose=1, mode='auto')
mp = 'C:/data/MC/best_male_female_{epoch:02d}-{val_loss:.4f}.hdf5'
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 32 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [es, lr])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))
# print(y_pred.shape)

# xy_train = train_datagen.flow_from_directory( #디렉토리채로 이미지 가져오기  //csv로 되있는걸 가져오는건 flow만 하면 됨
#     '../data/image/brain/train',#1. 경로
#     target_size = (150, 150), #2. 아직증폭안됨 = 타겟사이즈(임의로 정해도 됨)    shape(80, 150, 150, 1) : 0~1사이로 들어가 있음  //y = 0(라벨은 0이지만 shape = (80, ))
#     batch_size = 160, #배치사이즈     = test : (60, 150, 150, 1)  // 
#     class_mode = 'binary', #모드 #y값은 앞에있는애는 0 뒤에있는애는 1 맞고 틀림 느낌
#     save_to_dir='../data/image/brain_generator/train' #정의해논걸 print로 한번 건드려줘야 작성함(건드려 준 만큼 이미지 생성됨)
# )

# loss :  0.4204222857952118
# acc :  0.9061059951782227
# [[9.9999607e-01]
#  [1.0586260e-03]
#  [9.9652594e-01]
#  [2.0438510e-03]
#  [3.4913483e-01]
#  [3.1745746e-03]
#  [2.9591713e-03]
#  [9.9917608e-01]
#  [6.3648433e-05]
#  [9.4725631e-02]]
# [1. 0. 1. 0. 0. 0. 0. 1. 0. 0.]
# 0

import matplotlib.pyplot as plt
plt.imshow(x_test[0]) # 1이 남자 0이 여자
plt.show()