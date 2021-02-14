#나를 찍어서 내가 남자인지 여자인지에 대해
#실습
#남자 여자 구분
#ImageDataGenerator의 fit사용해서 완성

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#http://naver.me/5B1Y91UT
#이미지 / 데이터로 전처리 / 증폭가능하게 //

#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255, #전처리 
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=[-1, 1],
    height_shift_range=[-1, 1],
    rotation_range=3,
    zoom_range=0.3,
    # shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255) 

# #여자 증폭
# xy_train1 = train_datagen.flow_from_directory(
#     'C:/data/image/gender/female',
#     target_size = (128, 128),
#     batch_size = 2000,
#     class_mode= 'binary' ,
#     save_to_dir='C:/data/image/gender_generator/train_image_1' #정의해논걸 print로 한번 건드려줘야 작성함(건드려 준 만큼 이미지 생성됨)
# )

# #남자 증폭
# xy_train = train_datagen.flow_from_directory(
#     'C:/data/image/gender/male',
#     target_size = (128, 128),
#     batch_size = 2000,
#     class_mode= 'binary' ,
#     save_to_dir='C:/data/image/gender_generator/train_image_2' #정의해논걸 print로 한번 건드려줘야 작성함(건드려 준 만큼 이미지 생성됨)
# )
# print(xy_train1[0][0])
# print(xy_train1[0][1])
# print(xy_train[0][0])
# print(xy_train[0][1])


#남자 증폭
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128),
    batch_size = 4000,
    class_mode= 'binary' ,
)

xy_test = test_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128),
    batch_size = 4000,
    class_mode= 'binary' ,
)

#new_test데이터 정제
new_test = test_datagen.flow_from_directory(
    'C:/data/image/gender_test',
    target_size = (128, 128),
    batch_size = 2000,
    class_mode= 'binary',
    save_to_dir='C:/data/image/gender_test_resize'
)

# print(train_datagen) #Found 5208 images belonging to 2 classes.
# print(test_datagen)


# #npy 저장
np.save('C:/data/image/gender_npy/keras67_train_x.npy', arr=xy_train[0][0]) #x 1
np.save('C:/data/image/gender_npy/keras67_train_y.npy', arr=xy_train[0][1])
np.save('C:/data/image/gender_npy/keras67_test_x.npy', arr=xy_test[0][0]) #x 1
np.save('C:/data/image/gender_npy/keras67_test_y.npy', arr=xy_test[0][1])
np.save('C:/data/image/gender_npy/keras67_new_test.npy', arr=new_test[0][0])


#npy 로드
x_train = np.load('C:/data/image/gender_npy/keras67_train_x.npy') #x 1
y_train = np.load('C:/data/image/gender_npy/keras67_train_y.npy')
x_test = np.load('C:/data/image/gender_npy/keras67_test_x.npy') #x 1
y_test = np.load('C:/data/image/gender_npy/keras67_test_y.npy')
new_test = np.load('C:/data/image/gender_npy/keras67_new_test.npy')
print(x_train.shape, y_train.shape) #(1736, 64, 64, 3) (1736,)
print(new_test.shape)

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
x = Conv2D(256, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, activation='relu')(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

model.save('C:/data/h5/male_female_model.h5')#모델저장
#model = load_model('C:/data/h5/male_female_model.h5')#모델로드

#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'acc', patience = 50, mode = 'auto')
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=20, verbose=1, mode='auto')
modelpath = 'C:/data/MC/best_male_female_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 32 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [es, lr, mc])

model.save('../data/h5/male_female_model2.h5')
model.save_weights('../data/h5/male_female__weight.h5')

#4. 평가
# model = load_model('C:/data/h5/male_female_model2.h5')
# model.load_weights('C:/data/h5/male_female__weight.h5')

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
results = model.predict(new_test)

# 남/여
if results > 0.5 :
    print("요청한 사진은 ", results*100, "% 확률로 남자입니다.")
else :
    print("요청한 사진은" , (1-results)*100, "% 확률로 여자입니다." )

# loss :  0.4449421763420105
# acc :  0.8974654674530029
# 내가 : 남자 :  [[96.09087]] %

# loss :  0.5212218761444092
# acc :  0.8652073740959167
# 김광석 : 남자일 확률 :  [[99.55093]] %

# loss :  0.2705157697200775
# acc :  0.9282833933830261
# 요청한 사진은  [[99.23718]] % 확률로 남자입니다.
