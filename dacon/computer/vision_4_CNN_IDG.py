import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
train = pd.read_csv('C:/STUDY/dacon/computer/train.csv')
test = pd.read_csv('C:/STUDY/dacon/computer/test.csv')

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)

x = train.drop(['id', 'digit', 'letter'], axis=1).values
x_pred = test.drop(['id', 'letter'], axis = 1).values

#이미지 보기 왜안보이냐
# plt.imshow(x[100].reshape(28,28))

x = x.reshape(-1, 28, 28, 1)
x = x/255
x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255

y = train['digit']

y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

skf = StratifiedKFold(n_splits= 5, random_state=77, shuffle=True)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state=0)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state=0)

#ImageDatagenrtator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1, 1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(x,train['digit']) :

    x_train = x[train_index]
    x_valid = x[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(x_pred,shuffle=False)


    #모델링 CNN
    inputs = Input(shape = (28,28,1))
    conv2d = Conv2D(28, (4), strides =1 ,padding = 'SAME', input_shape = (28,28,1))(inputs)
    conv2d = Conv2D(64, (2), strides =1, padding = 'SAME', activation='relu')(conv2d)
    conv2d = Conv2D(64, (2), strides =1, padding = 'SAME', activation='relu')(conv2d)
    mp = MaxPooling2D(2)(conv2d)
    dp = Dropout(0.2)(mp)

    conv2d = Conv2D(32, (2), strides =1, padding = 'SAME', activation='relu')(dp)
    conv2d = Conv2D(32, (2), strides =1, padding = 'SAME', activation='relu')(conv2d)
    conv2d = Conv2D(32, (2), strides =1, padding = 'SAME', activation='relu')(conv2d)
    conv2d = Conv2D(64, (2), strides =1, padding = 'SAME', activation='relu')(conv2d)
    mp = MaxPooling2D(2)(conv2d)
    dp = Dropout(0.2)(mp)
    flt = Flatten()(dp)

    dense = Dense(32, activation='relu')(flt)
    dense = Dense(32, activation='relu')(dense)
    dense = Dense(16, activation='relu')(dense)

    outputs = Dense(10, activation='softmax')(dense) #분류 수 만큼 output
    model = Model(inputs = inputs, outputs = outputs)


# model.save('C:/data/h5/vision_model1.h5')#모델저장
# model = load_model('C:/data/h5/vision_model1.h5')#모델로드


#3. 훈련
    from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    lr1 = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')
    es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
    modelpath = 'C:/data/MC/best_cvision_{epoch:02d}-{val_loss:.4f}.hdf5'
    mc = ModelCheckpoint(filepath = modelpath ,save_best_only=True, mode = 'auto')
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001,epsilon=None) , metrics = ['acc'])
    learning_hist = model.fit(train_generator, epochs = 100, batch_size = 64, validation_data=(valid_generator), verbose = 1 ,callbacks = [es, lr1]) #mc
    #train_genrtator: x_train, y_train
    #test_generator: x_test, y_test
    
    # predict
    # model.save('C:/data/h5/vision_model2.h5') #모델저장2
    # model.save_weights('C:/data/h5/vision_model2_weight.h5') #weight저장
    # model.load_model('C:/data/h5/vision_model2.h5') #모델불러오기
    # model.load_weights('C:/data/h5/vision_model2_weight.h5') #weight불러오기
    result += model.predict_generator(test_generator,verbose=True)/5

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())

    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

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
submission.to_csv('C:/STUDY/dacon/computer/2021.02.03.csv',index=False)

'''
sub['digit'] = result.argmax(1)

submission = pd.read_csv('C:/STUDY/dacon/computer/submission.csv')
submission['digit'] = np.argmax(model.predict(cnn_test), axis=1)
submission.head()

submission.to_csv('C:/STUDY/dacon/computer/2021.02.02.csv', index=False)
'''
