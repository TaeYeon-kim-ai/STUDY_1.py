import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout, GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
#1. DATA
#.npy Load
x = np.load('C:/data/npy/train_x_128.npy', allow_pickle=True)
y = np.load('C:/data/npy/train_y_128.npy', allow_pickle=True)
target = np.load('C:/data/npy/target_128.npy', allow_pickle=True)

#generagtor
from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
target = preprocess_input(target)

idg = ImageDataGenerator(
    zoom_range = 0.2, #zoom_range = 0.1 X , 
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1),
    rotation_range=64
    # fill_mode='nearest',
)

idg2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 66, shuffle = True)

#control    
image_size = (128, 128, 3)
bts = 16
optimizer = Adam(learning_rate = 0.002)

train_generator = idg.flow(x_train, y_train, batch_size = bts, seed= 2028)
valid_generator = idg2.flow(x_val, y_val)

#2. MODEL
from tensorflow.keras.applications import EfficientNetB3
TF = EfficientNetB3(weights="imagenet", include_top=False, input_shape = image_size)    
TF.trainable = True
x = TF.output

x = Conv2D(2048, 4, padding="SAME", activation='swish')(x)
x = GlobalAveragePooling2D()(x)

x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = GaussianDropout(0.3)(x)



outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs = TF.input, outputs = outputs)
model.summary()

#COMPILE   
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_1.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
model.fit_generator(train_generator, epochs = 100 ,steps_per_epoch=len(x_train)/16, validation_data= valid_generator, callbacks=[es, rl, mc])

model.save('C:/data/h5/LT_vision_model2_5.h5')
model.save_weights('C:/data/h5/LT_vision_5.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#EVAL
model.load_weights('C:/data/MC/best_LT_vision2_1.hdf5')
result = model.predict(target,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred2.csv',index=False)
