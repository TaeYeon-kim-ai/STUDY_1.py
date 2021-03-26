import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt
import datetime

from scipy import stats

from sklearn.model_selection import KFold, train_test_split

from tensorflow.keras.applications import MobileNet, EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,\
    BatchNormalization, Activation, Dense, Dropout, Input, Concatenate, \
        GlobalAveragePooling2D, GaussianDropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import train

str_time = datetime.datetime.now()

kf = KFold(
    n_splits=5, shuffle=True
)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=(-1, 1),
    height_shift_range=(-1, 1)
)

datagen2 = ImageDataGenerator()

es = EarlyStopping(
    patience=20,
    verbose=1
)

rl = ReduceLROnPlateau(
    patience=10,
    verbose=1
)

x = np.load(
    '../../data/npy/train_x_128.npy'
)

y = np.load(
    '../../data/npy/train_y_128.npy'
)

test = np.load(
    '../../data/npy/predict_x_128.npy'
)

submission = pd.read_csv(
    'C:/data/LPD_competition/sample.csv'
)

# mob = MobileNet(
#     include_top=False,
#     input_shape=(128, 128, 3)
# )

eff = EfficientNetB4(
    include_top=False,
    input_shape=(128, 128, 3)
)

# mob.trainable = True
eff.trainable = True

batch_size = 16
epochs = len(x)//batch_size

x = preprocess_input(x)
test = preprocess_input(test)

count = 0
results = 0
for train_index, val_index in kf.split(x, y):
    print(str(count) + ' 번째 훈련 시작')

    x_train = x[train_index]
    x_val = x[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    # train_set = datagen.flow(
    #     x_train, y_train,
    #     batch_size = batch_size
    # )

    # val_set = datagen.flow(
    #     x_val, y_val,
    #     batch_size = batch_size
    # )

    mc = ModelCheckpoint(
        'C:/data/MC/best_LT_vision2_LT_' + str(count) + '.hdf5',
        save_best_only=True,
        verbose=1
    )
    image_size = (128, 128, 3)
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

    model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics='acc'
    )

    # hist = model.fit_generator(
    #     train_set,
    #     validation_data=val_set,
    #     epochs=1,
    #     steps_per_epoch=2400,
    #     callbacks=[es, rl, mc]
    # )

    model.fit(
        x_train, y_train,
        validation_data = (x_val, y_val),
        epochs = 1,
        batch_size = 16,
        callbacks = [es, rl, mc]
    )

    model.load_weights(
        'C:/data/MC/best_LT_vision2_LT_' + str(count) + '.hdf5'
    )

    pred = model.predict(
        test
    )

    results += np.argmax(pred, axis = -1)/5
    
    submission['prediction'] = np.argmax(pred, axis=-1)
    submission.to_csv(
        'C:/data/LPD_competition/pred' + str(count) + '.csv',
        index = False
    )

    print(str(count) + ' 번째 훈련 종료')
    print('time : ', datetime.datetime.now() - str_time)

    count += 1

submission['prediction'] = np.argmax(results, axis = -1)
submission.to_csv(
    'C:/data/LPD_competition/pred_Fi.csv',
    index = False
)

a = stats.mode(results, axis = 1).mode
a = np.argmax(a, axis = -1)

submission['prediction'] = a
submission.to_csv(
    'C:/data/LPD_competition/pred_Fi2.csv',
    index = False
)


print('time : ', datetime.datetime.now() - str_time)
print('done')
