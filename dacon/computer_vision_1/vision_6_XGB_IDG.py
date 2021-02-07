import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from keras import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from xgboost import XGBClassifier
import warnings

from tensorflow.python.eager.monitoring import Sampler
warnings.filterwarnings("ignore")

#1. 데이터
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
test = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)

# print(train['digit'].value_counts())
# 2    233
# 5    225
# 6    212
# 4    207
# 3    205
# 1    202
# 9    197
# 7    194
# 0    191
# 8    182


x = train.drop(['id', 'digit', 'letter'], axis=1).values
x_pred = test.drop(['id', 'letter'], axis = 1).values


# #이미지 보기 # 3
# plt.imshow(x[20].reshape(28,28)) # 3
# plt.show()

#데이터 reshape
x = x.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)

x = x/255.
x_pred = x_pred/255.

y = train['digit'] # 숨겨진 숫자 값


#ImageDatagenrtator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()


# sample_data = x[100].copy()
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

kfold = KFold(n_splits= 5, shuffle = True)

for train_index, test_index in kfold.split(x): 
    # print("TRAIN:", train_index, "TEST:", test_index) 
    x_train, x_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    x_train, y_train = idg.flow(x_train, y_train)
    x_test, y_test = idg2.flow(x_test, y_test)
    test_generator = idg2.flow(x_pred,shuffle=False)

    #모델링
    model = XGBClassifier( 
        learning_rate = 0.01,
        n_jobs = -1)

    #훈련
    learning_hist = model.fit(x_train, y_train)

    # predict
    # model.save('C:/data/h5/vision_model2.h5') #모델저장2
    # model.save_weights('C:/data/h5/vision_model2_weight.h5') #weight저장
    # model.load_model('C:/data/h5/vision_model2.h5') #모델불러오기
    # model.load_weights('C:/data/h5/vision_model2_weight.h5') #weight불러오기
    result = model.predict_generator(test_generator,verbose=True)/8

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)

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
submission.to_csv('C:/STUDY/dacon/computer/2021.02.04.csv',index=False)