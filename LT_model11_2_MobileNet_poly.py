import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout 
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd

x = np.load('../../data/npy/train_x_128.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_128.npy', allow_pickle=True)
target = np.load('../../data/npy/predict_x_128.npy', allow_pickle=True)

print(x.shape)
#x = x.reshape(48000, 192*192*3)

#안됨
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree = 2)
# poly.fit_transform(x)
# #교호작용 변수만 만들기   
# #poly_d2 = PolynomialFeatures(degree = 2, interaction_only=True)

#x = x.reshape(48000, 192, 192, 3)

#generagtor
idg = ImageDataGenerator(
    rescale= 1/255.,
    zoom_range = 0.3,
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1),
    rotation_range=40, 
    fill_mode='nearest'
)

idg2 = ImageDataGenerator(rescale= 1/255.)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 66, shuffle = True)

#control
image_size = (128, 128, 3)
bts = 32
optimizer = Adam(learning_rate = 0.001)

train_generator = idg.flow(x_train, y_train, batch_size = bts)
valid_generator = idg2.flow(x_val, y_val)
test_generator = idg2.flow(target)

#2. MODEL
from tensorflow.keras.applications import MobileNet
TF = MobileNet(weights="imagenet", include_top=False, input_shape = image_size)    
TF.trainable = True
x = TF.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs = TF.input, outputs = outputs)
model.summary()

#COMPILE   
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.train import Checkpoint, latest_checkpoint
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_LT.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
model.fit_generator(train_generator, epochs=5, verbose=1, validation_data= valid_generator, callbacks=[es, rl, mc])

model.save('C:/data/h5/LT_vision_6.h5')
model.save_weights('C:/data/h5/LT_vision_model2_6.h5')
# model = load_model('C:/data/h5/LT_vision_model2_5_mobileNet.h5')
# model.load_weights('C:/data/h5/LT_vision_5_mobileNet.h5')

latest_checkpoint('C:/data/MC/')

#EVAL
loss, acc = model.evaluate(valid_generator)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(test_generator,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred2.csv',index=False)