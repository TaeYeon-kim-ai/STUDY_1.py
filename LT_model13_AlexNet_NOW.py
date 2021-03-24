

#ImageNet Classification with Deep Convolutional Neural Networks
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout , Activation, Softmax, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import tensorflow as tf

print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

x = np.load('../../data/npy/train_x_224_40.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_224_40.npy', allow_pickle=True)
target = np.load('../../data/npy/predict_x_224_40.npy', allow_pickle=True)

print(x.shape)
print(y.shape)
print(target.shape)

#generagtor
from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
target = preprocess_input(target)

idg = ImageDataGenerator(
    zoom_range = 0.2,
    height_shift_range=(-1,1),
    width_shift_range=(-1,1),
    rotation_range=40)

idg2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.9, random_state = 66, shuffle = True)

#control
bts = 32
optimizer = Adam(learning_rate = 0.001)

train_generator = idg.flow(x_train, y_train, batch_size = bts)
valid_generator = idg2.flow(x_val, y_val)
test_generator = idg2.flow(target)

#2. MODEL
model = Sequential()
img_shape = (224,224,3)
no_of_classes=1000
# 레이어 1
model.add(Conv2D(96, (11,11), input_shape=img_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 레이어 2
model.add(Conv2D(256, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 레이어 3
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(384, (3,3), padding='same'))
model.add(Activation('relu'))

# 레이어 4
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(384, (3,3), padding='same'))
model.add(Activation('relu'))

# 레이어 5
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 레이어 6
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 레이어 7
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 레이어 8
model.add(Dense(no_of_classes))
model.add(Activation('softmax'))
model.summary()

#COMPILE   
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.train import Checkpoint, latest_checkpoint
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_LT_AlexNet.hdf5', save_best_only=True, mode = 'auto')
#cp = latest_checkpoint('C:/data/MC/')
es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='auto')
model.fit(train_generator, epochs=100, verbose=1, steps_per_epoch=x_train[-1]/32,validation_data= valid_generator, callbacks=[es, rl, mc])

model.save('C:/data/h5/LT_vision_AlexNet.h5')
model.save_weights('C:/data/h5/LT_vision_model2_AlexNet.h5')
# model = load_model('C:/data/h5/LT_vision_model2_5_mobileNet.h5')
# model.load_weights('C:/data/h5/LT_vision_5_mobileNet.h5')

#EVAL
loss, acc = model.evaluate(valid_generator)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(test_generator,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred_AlexNet.csv',index=False)
