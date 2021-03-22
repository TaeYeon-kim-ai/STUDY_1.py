import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, AveragePooling2D, AvgPool2D, Dropout, Flatten, BatchNormalization


#http://naver.me/5B1Y91UT
#이미지 / 데이터로 전처리 / 증폭가능하게 //

#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255, #전처리 
    horizontal_flip=True,#수평방향 뒤집기
    vertical_flip=True,#수직방향 뒤집기
    width_shift_range=0.1,#옆으로 이동
    height_shift_range=0.1,#위아래로 이동
    rotation_range=5,#지정된 각도 범위내에서 임의로 원본 이미지를 회전 시킵니다. 단위 도, 정수형 ex>rotaion_range=90
    zoom_range=1.2,#(1 - 수치) ~ (1+ 수치) 사이의 범위로 확대 축소를 합니다.
    shear_range=0.7,#밀린강도 범위내에서 임의로 원본 이미지를 변형 시킵니다. 수치는 시계반대방향으로 밀림강도를 radian으로 나타냅니다. 
    fill_mode='nearest'#가장 최근에 했던거와 유시한것 이동하면 이동전의 공간이 nall로 되는데 그 이동한 자리를 유사한 수치로 채우겠다.
                       #padding개념과 같음 0으로 주면 패딩하나생기는 느낌
)

test_datagen = ImageDataGenerator(rescale=1./255) #전처리만 함 0~1사이로 // 시험문제를 건드릴 필요는 없음.

#C:\data\image\brain\test\ad 걸렸다 #100,100,3 ad
#                        \normal 일반 #
#C:\data\image\brain\train\ad 걸렸다
#                        \normal 일반


# **flow 또는 flow_from_directory    #이미지 데이터 제너레이터를 통해 증폭하고,  flow로 수치화하고 데이터화 # tensorflow시험 3번문제
xy_train = train_datagen.flow_from_directory( #디렉토리채로 이미지 가져오기  //csv로 되있는걸 가져오는건 flow만 하면 됨
    '../data/image/brain/train',#1. 경로
    target_size = (64, 64), #2. 아직증폭안됨 = 타겟사이즈(임의로 정해도 됨)    shape(80, 150, 150, 1) : 0~1사이로 들어가 있음  //y = 0(라벨은 0이지만 shape = (80, ))
    batch_size = 5, #배치사이즈     = test : (60, 150, 150, 1)  // 
    class_mode = 'binary' #모드 #y값은 앞에있는애는 0 뒤에있는애는 1 맞고 틀림 느낌
)
# Found 160 images belonging to 2 classes.


#train_generator

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory( #디렉토리채로 이미지 가져오기  //csv로 되있는걸 가져오는건 flow만 하면 됨 
    '../data/image/brain/test',#1. 경로
    target_size = (64, 64), #2. 아직증폭안됨 = 타겟사이즈(임의로 정해도 됨)    shape(80, 150, 150, 1) : 0~1사이로 들어가 있음  //y = 0(라벨은 0이지만 shape = (80, ))
    batch_size = 5, #배치사이즈     = test : (60, 150, 150, 1)  // 
    class_mode = 'binary' #모드 #y값은 앞에있는애는 0 뒤에있는애는 1 맞고 틀림 느낌 #폴더구조 자체로 라벨링이 먹힌다.
)
# Found 120 images belonging to 2 classes.

#모델링
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = (64, 64, 3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(MaxPool2D(2))
model.add(Conv2D(128, (3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit_generator(xy_train, 
    steps_per_epoch=32, 
    epochs=50, 
    validation_data=xy_test, 
    validation_steps=4
    )


#평가
loss = model.evaluate(xy_test)
print('loss : ', loss)

#시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#시각화 할 것
import matplotlib.pyplot as plt
print("acc : ", acc[-1])
print("val_acc : ", val_acc[-1])

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




