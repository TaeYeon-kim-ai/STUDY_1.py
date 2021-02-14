# 나를 찍어서 남자인지 여자인지 구별
# acc도 나오게끔

# fit_generator 사용

import tensorflow
import numpy as np

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# male = 841
# female = 895

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    vertical_flip=True,
    horizontal_flip=True,
    zoom_range=0.7,
    rescale=1./255,
    validation_split=0.2,
    shear_range=1.2
)

datagen2=ImageDataGenerator(rescale=1./255)

train=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset="training"
)

val=datagen.flow_from_directory(
    'c:/data/image/data/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=32,
    subset="validation"
)

im=Image.open(
    'c:/data/image/data/my.jpg'
)

my=np.asarray(im)
my=np.resize(
    my,
    (128, 128, 3)
)
my=my.reshape(
    1, 128, 128, 3
)
predict=datagen2.flow(my)

# print(male[0][1])
# print(male2[0])


# female=datagen.flow_from_directory(
#     'c:/data/image/data/',
#     target_size=(128, 128),
#     class_mode='binary',
#     batch_size=32
# )

# female_generator=datagen2.flow_from_directory(
#     'c:/data/image/data/female',
#     target_size=(128, 128),
#     class_mode='binary',
#     batch_size=32
# )


# train, test=train_test_split(
#     train,
#     train_size=0.8,
#     random_state=32
# )

# print(train[0])


model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.1),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')

history = model.fit_generator(
    train,
    steps_per_epoch=44,
    epochs=5,
    callbacks=[es, rl],
    validation_data=val
)

pred=np.where(model.predict(predict)>0.5, 1, 0)
print(pred)
print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

if pred==1:
    print('남성')

else:
    print('여성')

# results