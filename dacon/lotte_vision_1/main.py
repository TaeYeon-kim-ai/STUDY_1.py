import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import  Dense, Flatten, Conv2D, Input, MaxPooling2D, LeakyReLU, Softmax, GlobalAvgPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
import pandas as pd


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
    rotation_range=32 
)

idg2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.9, random_state = 128, shuffle = True)

#control
bts = 128 
optimizer = Adam(learning_rate = 1e-3)

train_generator = idg.flow(x_train, y_train, batch_size = bts, seed=2048)
valid_generator = idg2.flow(x_val, y_val)
test_generator = idg2.flow(target)

# #2. MODEL

# model.save('C:/data/h5/LT_vision_model2_1.h5')
# model.save_weights('C:/data/h5/LT_vision_1.h5')
model = load_model('C:/data/h5/LT_vision_model2_1.h5')
model.load_weights('C:/data/h5/LT_vision_1.h5')

#EVAL
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)

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