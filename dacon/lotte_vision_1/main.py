import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAveragePooling2D, BatchNormalization, Dropout , GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
#1. DATA
#.npy Load
x = np.load('../../data/npy/train_x_256.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_256.npy', allow_pickle=True)
x_pred = np.load('../../data/npy/predict_x_256.npy', allow_pickle=True)

from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.9, random_state = 128, shuffle = True)

#control
image_size = (256, 256, 3)
bts = 32
optimizer = Adam(learning_rate = 0.001)

#2. MODEL
model = load_model('C:/data/h5/LT_vision_model2_7.h5')
model.load_weights('C:/data/h5/LT_vision_7.h5')

#EVAL
loss, acc = model1.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(x_pred,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred_21.03.24_1.csv',index=False)

