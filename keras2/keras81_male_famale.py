# VGG16모델로 남녀구분 하기
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, AveragePooling2D, AvgPool2D, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2


#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255) 

#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128), 
    batch_size = 32, 
    class_mode = 'binary',
    subset="training"
)

xy_val = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128), 
    batch_size = 32, 
    class_mode = 'binary',
    subset="validation"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/gender',
    target_size = (128, 128),
    batch_size = 32, 
    class_mode = 'binary'
)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
# TF = VGG16(weights= 'imagenet', include_top= False, input_shape= (128, 128, 3))
# TF.trainable = False
# TF_model = TF
# last = TF_model.output

# x = Flatten()(last)

# x = Dense(512, activation='relu')(x)
# x = BatchNormalization()(x)
# outputs = Dense(1, activation='sigmoid')(x)
# model = Model(inputs = TF_model.input, outputs = outputs)
# model.summary()

# model.compile(
#     loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['acc'])
# es=EarlyStopping(patience=10, verbose=1, monitor='loss')
# rl=ReduceLROnPlateau(patience=5, verbose=1, monitor='loss')
# hist= model.fit_generator(xy_train, steps_per_epoch=44, epochs=100, validation_data=xy_val, callbacks=[es, rl])

#model.save('../data/h5/male_female_model2.h5')
#model.save_weights('../data/h5/male_female__weight.h5')
model = load_model('../data/h5/male_female_model2.h5')
model.load_weights('../data/h5/male_female__weight.h5')

#평가
loss, acc = model.evaluate(xy_test)
print('loss : ', loss)
print('acc : ', acc)

#resize
def Dataization(img_path):
    image_w = 128
    image_h = 128
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)
 
src = []
name = []
test = []
image_dir = "C:/data/image/gender_test/"
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
y_pred = model.predict(test)
 
for i in range(len(test)):
    if y_pred[i] > 0.5 :
        print(name[i] + y_pred[i]*100, "% 확률로 남자입니다..")
    else :
        print(name[i] + (1-y_pred[i])*100, "% 확률로 여자입니다." )

#시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#그래프 출력
import matplotlib.pyplot as plt
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

# 그래프출력(활성화함수)
def softstep_func(x): # Soft step (= Logistic), 시그모이드(Sigmoid, S자모양) 대표적인 함수
    return 1 / (1 + np.exp(-x))
 
#그래프 출력
plt.plot(x, softstep_func(x), linestyle='--', label="Soft step (= Logistic)")