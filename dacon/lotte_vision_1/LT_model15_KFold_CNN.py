# kfold

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
    'c:/data/npy/lotte_xs.npy'
)

y = np.load(
    'c:/data/npy/lotte_ys.npy'
)

test = np.load(
    'c:/data/npy/lotte_tests.npy'
)

submission = pd.read_csv(
    'c:/LPD_competition/sample.csv'
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
        'c:/data/modelcheckpoint/lotte_' + str(count) + '.hdf5',
        save_best_only=True,
        verbose=1
    )

    model = Sequential()
    model.add(eff)
    # model.add(Conv2D(1024, kernel_size=3, padding='same', activation = 'swish'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='swish'))
    model.add(GaussianDropout(0.4))
    model.add(Dense(1000, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
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
        epochs = 100,
        batch_size = 16,
        callbacks = [es, rl, mc]
    )

    model.load_weights(
        'c:/data/modelcheckpoint/lotte_' + str(count) + '.hdf5'
    )

    pred = model.predict(
        test
    )

    results += np.argmax(pred, axis = -1)/5
    
    submission['prediction'] = np.argmax(pred, axis=-1)
    submission.to_csv(
        'c:/data/csv/lotte_' + str(count) + '.csv',
        index = False
    )

    print(str(count) + ' 번째 훈련 종료')
    print('time : ', datetime.datetime.now() - str_time)

    count += 1

submission['prediction'] = np.argmax(results, axis = -1)
submission.to_csv(
    'c:/data/csv/lotte.csv',
    index = False
)

a = stats.mode(results, axis = 1).mode
a = np.argmax(a, axis = -1)

submission['prediction'] = a
submission.to_csv(
    'c:/data/csv/lotte_s.csv',
    index = False
)


print('time : ', datetime.datetime.now() - str_time)
print('done')