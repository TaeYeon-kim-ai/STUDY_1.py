#롯데 데이터셋
#.npy

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. data
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    zoom_range = 0.1,
    height_shift_range=0.04,
    width_shift_range=0.04,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale = 1./255)

#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/LPD_competition/train',
    target_size = (256, 256),
    batch_size = 100,
    class_mode = 'categorical',
    shuffle = False,
    subset="training"
)

xy_val = train_datagen.flow_from_directory(
    'C:/data/LPD_competition/train', # same directory as training data0
    target_size=(256, 256),
    batch_size=100,
    class_mode='categorical',
    shuffle = False,
    subset='validation') # set as validation data

x_test = train_datagen.flow_from_directory(
    'C:/data/LPD_competition/test',
    target_size = (256, 256),
    shuffle = False
)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB5, MobileNet

# tf.keras.applications.MobileNet(
#     input_shape=None,
#     alpha=1.0,
#     depth_multiplier=1,
#     dropout=0.001,
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     **kwargs
# )

mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

top_model = mobile_net.output
top_model = Flatten()(top_model)
top_model = Dense(512, activation="relu")(top_model)
top_model = Dense(1000, activation="softmax")(top_model)

model = Model(inputs=mobile_net.input, outputs = top_model)

model.compile(
    loss = 'categorical_crossentropy', 
    optimizer=Adam(learning_rate = 0.2),
    metrics=['acc']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience=10, verbose=1, monitor='val_loss')
rl = ReduceLROnPlateau(patience=5, verbose=1, monitor='val_loss')
history = model.fit_generator(
    # steps_per_epoch 는 전체 데이터를 에포치로 나눈 값을 넣는다.
    xy_train, steps_per_epoch=len(xy_train), 
    epochs=50, 
    validation_data=xy_val, 
    validation_steps=(len(xy_val))
)


# 평가
loss, acc = model.evaluate(x_test)
print("loss : ", loss)
print("acc : ", acc)

acc = history.history['acc']
loss = history.history['loss']

# 시각화 할 것!!

print('acc :', acc[-1])

result = pd.read_csv("C:/data/LPD_competition/sample.csv")

# prd = model.predict(x_test)
# filenames = xy_test.filenames
# nb_samples = len(filenames)
# print(nb_samples)
prd = model.predict_generator(xy_test, steps=72000)
a = pd.DataFrame()
prd = pd.Series(np.argmax(prd,axis=-1))
prd = pd.concat([a,prd],axis=1)
result.iloc[:,1] = prd.sort_index().values
result.to_csv('C:/data/LPD_competition/sample_1.csv')


