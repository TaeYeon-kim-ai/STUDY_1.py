import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import  Sequential, Model
from tensorflow.keras.layers import  Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, AveragePooling2D
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import  Adam
import string
import pandas as pd
import cv2


#train, test 준비
# #1. data
#npy 로드
x = np.load('C:/data/vision_2/dirty_mnist_npy/x_train.npy') #x 1
x_test = np.load('C:/data/vision_2/dirty_mnist_npy/x_test.npy') #x 1
dataset = pd.read_csv('C:/data/vision_2/dirty_mnist_2nd_answer.csv')
submission = pd.read_csv('C:/data/vision_2/sample_submission.csv')
y = dataset.iloc[:,:]

print(x.shape)
print(x_test.shape)


import matplotlib.pyplot as plt

# print(y.shape)
#print(y_data)

#석성훈님
# plt.figure(figsize=(20, 5))
# ax = plt.subplot(2, 10, 1)
# plt.imshow(x_test[0])


# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.show()

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, shuffle= False, random_state=64)
print(x_train.shape, y_train.shape)


#costum

#mdeling
def model():
    inputs = Input(shape=(128, 128, 1)) 
    x = Conv2D(128, 4, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(128, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 2, activation='relu')(x)
    x = AveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation= 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'binary_crossentropy', optimizer=optimizers, metrics=['acc'])    
    model.summary()
    return model

# model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드

#3훈련

optimizers = Adam(lr=0.001,epsilon=None)
model = model()
mc = ModelCheckpoint('C:/data/MC/best_cvision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='loss', patience = 20, mode='auto')
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='auto')
model.fit(x_train, y_train, epochs = 1, batch_size = 64 ,validation_data = (x_val, y_val), callbacks=[es, lr, mc])

y_pred = model.predict(x_test)
print(y_pred)
y_pred_sub = np.where(y_pred < 0.5, 0, 1)
print(y_pred_sub)
submission[i] = y_pred_sub
    
submission.to_csv('C:/data/vision_2/submission_2021.02.13_z.csv', index=False)


print('예쁘게 돌아가자아~~~')
print('예쁘게 돌아가자아~~~')
print('예쁘게 돌아가자아~~~')
print('예쁘게 돌아가자아~~~')
print('예쁘게 돌아가자아~~~')
