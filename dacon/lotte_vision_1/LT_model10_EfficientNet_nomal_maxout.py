
class Maxout(Function):
    
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        max_out=4    #Maxout Parameter
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x= x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output    
        return input


#=================================================

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
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.core import max
TF = EfficientNetB5(weights="imagenet", include_top=False, input_shape = image_size)    
TF.trainable = True
x = TF.output
x = Conv2D(96, 2, padding='SAME', activation='swish')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(256, 2, padding='SAME', activation='swish')(x)
x = MaxPooling2D(2)(x)
x = GaussianDropout(rate=0.2)(x)

x = Conv2D(384, 2, padding='SAME', activation='swish')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(128, 2, padding='SAME', activation='swish')(x)
x = GlobalAveragePooling2D()(x)
x = GaussianDropout(rate=0.2)(x)

x = Flatten()(x)
x = Dense(2048, activation='swish')(x)
x = GaussianDropout(rate=0.2)(x)

outputs = Dense(1000)(x)
outputs = Maxout()(outputs)
model = Model(inputs = TF.input, outputs = outputs)
model.summary()

#COMPILE   
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['acc'])
mc = ModelCheckpoint('C:/data/MC/best_LT_vision2_{epoch:02d}-{val_loss:.4f}.hdf5', save_best_only=True, mode = 'auto')
es = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
model.fit(x_train, y_train, epochs=100, verbose=1, validation_data= (x_val, y_val), callbacks=[es, rl, mc])

model.save('C:/data/h5/LT_vision_model2_7.h5')
model.save_weights('C:/data/h5/LT_vision_7.h5')
# model = load_model('C:/data/h5/fish_model2.h5')
# model.load_weights('C:/data/h5/fish_weight.h5')

#EVAL
loss, acc = model.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)
result = model.predict(test_generator,verbose=True)

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/pred_21.03.24_1.csv',index=False)

