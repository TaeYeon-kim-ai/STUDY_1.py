import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout 
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet101
import pandas as pd
#1. DATA
#.npy Load
x = np.load('../../data/npy/LPD_train_x1.npy', allow_pickle=True)
y = np.load('../../data/npy/LPD_train_y1.npy', allow_pickle=True)
target = np.load('../../data/npy/target1.npy', allow_pickle=True)


def create_model():
    inputs = Input(shape = (128, 128, 3))    
    x = Conv2D(128, 3, padding="SAME", activation='relu', name = 'hiiden1')(inputs)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, 2, padding="SAME", activation='relu', name = 'hiiden2')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, 2, padding="SAME", activation='relu', name = 'hiiden3')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 2, padding="SAME", activation='relu', name = 'hiiden4')(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, 2, padding="SAME", activation='relu', name = 'hiiden5')(x)
    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)

    x = Dense(2048, activation='relu', name = 'hiiden6')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1000, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    return model

#KFold
from sklearn.model_selection import StratifiedKFold
kf =  StratifiedKFold(K, True, 7)
es_pationce = 7
rl_pationce = 3
epochs = 100
bts = 32

all_scores = []
all_preds = []


for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_generator = idg.flow(rescale=1/255., x_train, y_train, batch_size = bts, seed=1024)
    test_generator = idg2.flow(target)

    model = create_model()

    #COMPILE   
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
    mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
    es = EarlyStopping(monitor='val_loss', patience=es_pationce, verbose=1, mode='auto')
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=rl_pationce, verbose=1, mode='auto')
    model.fit_generator(train_generator, epochs=epochs, verbose=1, validation_data= valid_generator, callbacks=[es, rl, mc])

    model.save('C:/data/h5/LT_vision_model2_2.h5')
    model.save_weights('C:/data/h5/LT_vision_2.h5')
    # model = load_model('C:/data/h5/fish_model2.h5')
    # model.load_weights('C:/data/h5/fish_weight.h5')

    score = model.evaluate(test_generator, batch_size=bts)
    all_score.append(score)

    result = pd.read_csv("C:/data/LPD_competition/sample.csv")

    # prd = model.predict(x_test)
    # filenames = xy_test.filenames
    # nb_samples = len(filenames)
    # print(nb_samples)
    prd = model.predict_generator(test_generator, steps=72000)
    a = pd.DataFrame()
    prd = pd.Series(np.argmax(prd,axis=-1))
    prd = pd.concat([a,prd],axis=1)
    result.iloc[:,1] = prd.sort_index().values
    result.to_csv('C:/data/LPD_competition/sample_2.csv')