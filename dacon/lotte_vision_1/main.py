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
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 128, shuffle = True)

#control
filenum = 18
image_size = (256, 256, 3)
bts = 32
optimizer = Adam(learning_rate = 0.001)

#2. MODEL
model = load_model('C:/data/h5/LT_vision_model2_9.h5')
model.load_weights('C:/data/h5/LT_vision_9.h5')

#EVAL
loss, acc = model.evaluate(x_val, y_val)
print("loss : ", loss)
print("acc : ", acc)
y_pred = model.predict(x_pred, verbose=True)

print('Accuracy without TTA:',np.mean((y_val==y_pred)))

sub = pd.read_csv('C:/data/LPD_competition/sample.csv')
# sub['prediction'] = np.argmax(result,axis = 1)
# sub.to_csv('C:/data/LPD_competition/pred_21.03.26_2.csv',index=False)



#ef4 : point : 63.72
save_folder = '../../data/LPD_competition'

#======================[ T T A ]==========================
cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(x_pred, steps = len(x_pred), verbose = 1) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)

    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    count_result.append(count)
    print(f'기록 : {count_result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)