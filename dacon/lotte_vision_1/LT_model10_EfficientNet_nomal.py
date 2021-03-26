import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout , GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
#1. DATA
#.npy Load
x = np.load('../../data/npy/train_x_192_2.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_192_2.npy', allow_pickle=True)
x_pred = np.load('../../data/npy/predict_x_192.npy', allow_pickle=True)

from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 128, shuffle = True)

#control
image_size = (192, 192, 3)
bts = 32
optimizer = Adam(learning_rate = 0.001)

#2. MODEL
from tensorflow.keras.applications import EfficientNetB4
TF = EfficientNetB4(weights="imagenet", include_top=False, input_shape = image_size)    
TF.trainable = True
x = TF.output
x = Conv2D(96, 2, padding='SAME', activation='swish')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(256, 2, padding='SAME', activation='swish')(x)
x = GlobalAveragePooling2D()(x)

x = Flatten()(x)
x = Dense(2048, activation='swish')(x)
x = GaussianDropout(rate=0.2)(x)

outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs = TF.input, outputs = outputs)
model.summary()

#COMPILE   
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
model.fit(x_train, y_train, epochs=200, verbose=1, validation_data= (x_val, y_val), callbacks=[es, rl, mc])

model.save('C:/data/h5/LT_vision_model2_10.h5')
model.save_weights('C:/data/h5/LT_vision_10.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#EVAL
loss, acc = model.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(test_generator,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred_21.03.26_1.csv',index=False)


#ef4 : point : 63.72  256
#ef4_2 : point : 67.12  256