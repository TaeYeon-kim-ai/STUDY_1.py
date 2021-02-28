import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.eager.monitoring import Sampler
warnings.filterwarnings("ignore")

#1. 데이터
train = pd.read_csv('C:/data/vision_2/mnist_data/train.csv')
test = pd.read_csv('C:/data/vision_2/mnist_data/test.csv')

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)

#print(train['digit'].value_counts())

data = train.drop(['id', 'digit'], axis=1).values
target = test.drop(['id'], axis = 1).values

data = data.reshape(-1, 28, 28, 1)
target = target.reshape(-1, 28, 28, 1)
data = data.astype('float32')/255.
target = target.astype('float32')/255.


#ImageDatagenrtator & data augmentation
idg = ImageDataGenerator(
    height_shift_range= (-1, 1), 
    width_shift_range= (1, -1),
    )
idg2 = ImageDataGenerator()

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
lr1 = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')
skf = StratifiedKFold(n_splits= 10, random_state=66, shuffle=True)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(data,train['digit']) :
    
    x_train = data[train_index]
    x_valid = data[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train, batch_size = 64)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(target, shuffle=False)


    #모델링 CNN
    inputs = Input(shape = (28,28,1))
    x = Conv2D(32, (4), strides =1 ,padding = 'SAME', kernel_initializer='he_normal', input_shape = (28,28,1))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3), strides =1, padding = 'SAME', activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(26, activation='softmax')(x) #분류 수 만큼 output
    model = Model(inputs = inputs, outputs = outputs)

model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드


#3. 훈련
    modelpath = 'C:/data/MC/best_cvision_{epoch:02d}-{val_loss:.4f}.hdf5'
    mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001,epsilon=None) , metrics = ['acc'])
    learning_hist = model.fit_generator(train_generator, epochs = 1000, validation_data=(valid_generator), verbose = 1 ,callbacks = [es, lr1]) #mc
    result += model.predict_generator(test_generator,verbose=True)/16

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())

    nth += 1
    print(nth, '번째 학습을 완료.')

model.summary()

#3.1 시각화
hist = pd.DataFrame(learning_hist.history)
hist['val_loss'].min

hist.columns
plt.title('Training and validation loss')
plt.xlabel('epochs')

plt.plot(hist['val_loss'])
plt.plot(hist['loss'])
plt.legend(['val_loss', 'loss'])

plt.figure()

plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Traning and validation accuracy')

plt.show()

# model.save('C:/data/h5/vision_model2.h5') #모델저장2
# model.save_weights('C:/data/h5/vision_model2_weight.h5') #weight저장
# model.load_model('C:/data/h5/vision_model2.h5') #모델불러오기
# model.load_weights('C:/data/h5/vision_model2_weight.h5') #weight불러오기


# #4. 평가, 예측
# submission = pd.read_csv('C:/STUDY_1.py/dacon/computer/submission.csv')
# submission['digit'] = result.argmax(1)
# submission.to_csv('C:/STUDY_1.py/dacon/computer/2021.02.05.csv',index=False)
'''
