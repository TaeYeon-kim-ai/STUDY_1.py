import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout, GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from tensorflow import train


x = np.load('../../data/npy/train_x_160.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_160.npy', allow_pickle=True)
x_pred = np.load('../../data/npy/predict_x_160.npy', allow_pickle=True)
sub = pd.read_csv('C:/data/LPD_competition/sample.csv')

from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

print(x.shape)

#control
#time
str_time = datetime.datetime.now()

kf =KFold(
    n_splits = 5, shuffle=True #Stratified
)

#generagtor
idg = ImageDataGenerator(
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1),
    rotation_range=32
)

idg2 = ImageDataGenerator()

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='auto')

image_size = (160, 160, 3)
batch_size = 32
epochs = len(x)//batch_size
optimizer = Adam(learning_rate = 0.001)

count = 0
results = 0

#2. MODEL
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.train import Checkpoint, latest_checkpoint

for train_index, val_index in kf.split(x, y.argmax(1)) :
    print(str(count) + ' 번째 훈련 시작')

    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    

    #Check
    mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_LT_' + str(count) + '.hdf5', save_best_only=True, mode = 'auto')

    #TEST_MODEL
    inputs = Input(shape = image_size )    
    x = Conv2D(128, 3, padding="SAME", activation='relu', name = 'hiiden1')(inputs)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 2, padding="SAME", activation='relu', name = 'hiiden2')(x)
    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu', name = 'hiiden3')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1000, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()

    #MODEL
    # TF = EfficientNetB4(weights="imagenet", include_top=False, input_shape = image_size)    
    # TF.trainable = True
    # x = TF.output
    # x = Conv2D(96, 2, padding='SAME', activation='swish')(x)
    # x = MaxPooling2D(2)(x)

    # x = Conv2D(256, 2, padding='SAME', activation='swish')(x)
    # x = MaxPooling2D(2)(x)

    # x = Conv2D(256, 2, padding='SAME', activation='swish')(x)
    # x = GlobalAveragePooling2D()(x)

    # x = Flatten()(x)
    # x = Dense(256, activation='swish')(x)
    # x = GaussianDropout(rate=0.4)(x)

    # outputs = Dense(1000, activation='softmax')(x)
    # model = Model(inputs = TF.input, outputs = outputs)


    #COMPILE   
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc']) #sparse_
    model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 1, batch_size = batch_size, callbacks = [es, rl, mc])

    model.load_weights('../../data/MC/best_LT_vision2_LT_' + str(count) + '.hdf5')

    #PRED
    result = model.predict(x_pred, verbose=True)
    
    results += np.argmax(x_pred, axis = -1)/5

    sub['prediction'] = np.argmax(result, axis = 1)
    sub.to_csv('C:/data/LPD_competition/pred' + str(count) + '.csv', index=False)

    print(str(count) + ' 번째 훈련 종료')
    print('time : ', datetime.datetime.now() - str_time)

    count += 1

sub['prediction'] = np.argmax(results, axis = 1)
sub.to_csv('C:/data/LPD_competition/pred_Fi.csv', index=False)

a = stats.mode(results, axis = 1).mode
a = np.argmax(a, axis = -1)

sub['prediction'] = a
sub.to_csv('C:/data/LPD_competition/pred_Fi2.csv',index = False)


print('time : ', datetime.datetime.now() - str_time)
print('done')
