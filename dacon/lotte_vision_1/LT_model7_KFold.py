import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAvgPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam

#1. DATA
# train_datagen = ImageDataGenerator(
#     rescale= 1./255,
#     zoom_range=0.2,
#     height_shift_range = 0.1,
#     width_shift_range = 0.1
# )

# pred_datagen = ImageDataGenerator(rescale = 1.255)

#======================================================
# xy_train = train_datagen.flow_from_directory(
#     'C:/data/LPD_competition/train',
#     target_size = (172, 172),
#     batch_size = 100000,
#     class_mode = 'categorical',
#     shuffle = False,
#     subset="training"
# )

# #.npy transe pred
# x_pred = pred_datagen.flow_from_directory(
#     'C:/data/LPD_competition/test',
#     target_size = (172, 172),    
#     batch_size = 100000,
#     shuffle = False
# )

#print(xy_train[0][0].shape) #x만 나옴
#print(xy_train[0][1].shape) # y [0. 1. 1. 1. 1.]

#.npy전환
# np.save('C:/data/LPD_competition/npy/LT_x_train_244.npy', arr = xy_train[0][0])
# np.save('C:/data/LPD_competition/npy/LT_y_train_244.npy', arr = xy_train[0][1])
# np.save('C:/data/LPD_competition/npy/LT_x_pred_244.npy', arr = x_pred[0][0])

#.npy Load
x = np.load('C:/data/LPD_competition/npy/LT_x_train.npy', allow_pickle=True)
y = np.load('C:/data/LPD_competition/npy/LT_y_train.npy', allow_pickle=True)
target = np.load('C:/data/LPD_competition/npy/LT_x_pred.npy', allow_pickle=True)

from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
target = preprocess_input(target)

# print(x.shape)
# print(y.shape)
# print(target.shape)

#generagtor
idg = ImageDataGenerator(
    zoom_range = 0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)

idg2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 128, shuffle = True)

#control
bts = 64
optimizer = Adam(learning_rate = 1e-3)

train_generator = idg.flow(x_train, y_train, batch_size = bts)
valid_generator = idg2.flow(x_val, y_val)
test_generator = idg2.flow(target)

#2. MODEL
inputs = Input(shape = (128, 128, 3))    
x = Conv2D(64, 4, padding="SAME", activation='relu')(inputs)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Conv2D(128, 2, padding="SAME", activation='relu')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Conv2D(64, 2, padding="SAME", activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, 2, padding="SAME", activation='relu')(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)

outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs = inputs, outputs = outputs)
model.summary()

#COMPILE
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto')
model.fit_generator(train_generator, epochs=100, verbose=1, validation_data= valid_generator, callbacks=[es, rl, mc])

#model.save('C:/data/h5/LT_vision_model2_1.h5')
#model.save_weights('C:/data/h5/LT_vision_1.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#EVAL
loss, acc = model.evaluate(test_generator)
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
prd = model.predict_generator(test_generator, steps=72000)
a = pd.DataFrame()
prd = pd.Series(np.argmax(prd,axis=-1))
prd = pd.concat([a,prd],axis=1)
result.iloc[:,1] = prd.sort_index().values
result.to_csv('C:/data/LPD_competition/sample_1.csv')
