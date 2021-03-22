#.npy Load
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout 
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
#1. DATA
#.npy Load
x = np.load('../../data/npy/train_x_192.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_192.npy', allow_pickle=True)
target = np.load('../../data/npy/predict_x_192.npy', allow_pickle=True)

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
    rotation_range=32 
)

idg2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.9, random_state = 128, shuffle = True)

#control
image_size = (192, 192, 3)
bts = 32
optimizer = Adam(learning_rate = 1e-3)

train_generator = idg.flow(x_train, y_train, batch_size = bts, seed=1024)
valid_generator = idg2.flow(x_val, y_val)
test_generator = idg2.flow(target)

# model.save('C:/data/h5/LT_vision_model2_5_mobileNet.h5')
# model.save_weights('C:/data/h5/LT_vision_5_mobileNet.h5')
model = load_model('C:/data/h5/LT_vision_model2_4.h5')
model.load_weights('C:/data/h5/LT_vision_4.h5')

#EVAL
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(test_generator,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred1.csv',index=False)

