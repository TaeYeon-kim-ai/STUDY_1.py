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

from tensorflow.python.eager.monitoring import Sampler
warnings.filterwarnings("ignore")

#1. 데이터
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
test = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)

#print(train['digit'].value_counts())

data = train.drop(['id', 'digit', 'letter'], axis=1).values
target = test.drop(['id', 'letter'], axis = 1).values


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
skf = StratifiedKFold(n_splits= 16, random_state=66, shuffle=True)

val_loss_min = []
result = 0
nth = 0


for train_index, valid_index in skf.split(data,train['digit']) :

    x_train = data[train_index]
    x_valid = data[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train, batch_size = 8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(target, shuffle=False)


    #모델링 CNN
    inputs = Input(shape = (28,28,1))
    conv2d = Conv2D(32, (4), strides =1 ,padding = 'SAME', input_shape = (28,28,1))(inputs)
    btcn = BatchNormalization()(conv2d)
    mp = MaxPooling2D(2)(conv2d)

    conv2d = Conv2D(32, (3), strides =1, padding = 'SAME', activation='relu')(mp)
    btcn = BatchNormalization()(conv2d)
    conv2d = Conv2D(32, (3), strides =1, padding = 'SAME', activation='relu')(btcn)
    mp = MaxPooling2D(2)(conv2d)
    drop = Dropout(0.3)(mp)

    btcn = BatchNormalization()(drop)
    conv2d = Conv2D(64, (3), strides =1, padding = 'SAME', activation='relu')(btcn)
    btcn = BatchNormalization()(conv2d)
    conv2d = Conv2D(32, (3), strides =1, padding = 'SAME', activation='relu')(btcn)
    btcn = BatchNormalization()(conv2d)
    conv2d = Conv2D(32, (3), strides =1, padding = 'SAME', activation='relu')(btcn)
    btcn = BatchNormalization()(conv2d)
    mp = MaxPooling2D(2)(conv2d)
    drop = Dropout(0.3)(mp)
    flt = Flatten()(drop)

    dense = Dense(32, activation='relu')(flt)
    btcn = BatchNormalization()(dense)
    dense = Dense(128, activation='relu')(btcn)
    btcn = BatchNormalization()(dense)
    dense = Dense(64, activation='relu')(btcn)
    btcn = BatchNormalization()(dense)

    outputs = Dense(10, activation='softmax')(btcn) #분류 수 만큼 output
    model = Model(inputs = inputs, outputs = outputs)


# model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드


#3. 훈련
    from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    modelpath = 'C:/data/MC/best_cvision_{epoch:02d}-{val_loss:.4f}.hdf5'
    mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001,epsilon=None) , metrics = ['acc'])
    learning_hist = model.fit_generator(train_generator, epochs = 1000, validation_data=(valid_generator), verbose = 1 ,callbacks = [es, lr1]) #mc
    
    # predict
    # model.save('C:/data/h5/vision_model2.h5') #모델저장2
    # model.save_weights('C:/data/h5/vision_model2_weight.h5') #weight저장
    # model.load_model('C:/data/h5/vision_model2.h5') #모델불러오기
    # model.load_weights('C:/data/h5/vision_model2_weight.h5') #weight불러오기
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


#4. 평가, 예측
submission = pd.read_csv('C:/STUDY/dacon/computer/submission.csv')
submission['digit'] = result.argmax(1)
submission.to_csv('C:/STUDY/dacon/computer/2021.02.05.csv',index=False)



#=================================================================
#이미지 보기 # 3
# plt.imshow(data[20].reshape(28,28)) # 3
# plt.show()


# sample_data = data[100].copy()
# sample = expand_dims(sample_data,0)
# sample_datagen = ImageDataGenerator(
#     height_shift_range=(-1,1), 
#     width_shift_range=(-1,1), 
#     rotation_range = 10,
#     horizontal_flip = True
#     )

# sample_generator = sample_datagen.flow(sample, batch_size=1)

# plt.figure(figsize = (16,10)) #그림출력

# for i in range(9) : 
#     plt.subplot(3,3,i+1) #3,3으로 첫번째부터 삽입
#     sample_batch = sample_generator.next()
#     sample_image=sample_batch[0]
#     plt.imshow(sample_image.reshape(28,28))

# plt.show()