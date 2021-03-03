import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, concatenate, Input, Flatten, Dense
from tensorflow.keras import Model
import warnings
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.callbacks import ReduceLROnPlateau
warnings.filterwarnings("ignore")

#폴더 경로를 설정해줍니다.
os.chdir('C:/data/motion/') 

#제공된 데이터들의 리스트를 확인합니다.
print(os.listdir())

train = pd.read_csv('C:/data/motion/train_df.csv')
submission = pd.read_csv('C:/data/motion/sample_submission.csv')

print(train.head(2))

print(train.shape)

#submission 파일 불러오기
print(submission.head(2))

#glob를 활용해 이미지의 경로들을 불러옵니다.
import glob
train_paths = glob.glob('C:/data/motion/train_imgs/*.jpg')
test_paths = glob.glob('C:/data/motion/test_imgs/*.jpg')
print(len(train_paths), len(test_paths))


# 시각화
plt.figure(figsize=(40,20))
count=1

for i in np.random.randint(0,len(train_paths),5):
    
    plt.subplot(5,1, count)
    
    img_sample_path = train_paths[i]
    img = Image.open(img_sample_path)
    img_np = np.array(img)

    keypoint = train.iloc[:,1:49] #위치키포인트 하나씩 확인
    keypoint_sample = keypoint.iloc[i, :]
    
    for j in range(0,len(keypoint.columns),2):
        plt.plot(keypoint_sample[j], keypoint_sample[j+1],'rx')
        plt.imshow(img_np)
    
    count += 1


train['path'] = train_paths



def trainGenerator():
    for i in range(len(train)):
        img = tf.io.read_file(train['path'][i]) # path(경로)를 통해 이미지 읽기
        img = tf.image.decode_jpeg(img, channels=3) # 경로를 통해 불러온 이미지를 tensor로 변환
        img = tf.image.resize(img, [360,640]) # 이미지 resize 
        target = train.iloc[:,1:49].iloc[i,:] # keypoint 뽑아주기
        
        yield (img, target)

#generator를 활용해 데이터셋 만들기        
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([360,640,3]),tf.TensorShape([48])))
train_dataset = train_dataset.batch(32).prefetch(1)



#2.========================================== 모델링

#간단한 CNN 모델을 적용합니다.
TF = EfficientNetB6(weights= 'imagenet', include_top = False, input_shape = (360, 640, 3)) #레이어 16개
TF.trainable = False #훈련시키지 않고 가중치만 가져오겠다.
model = Sequential()
model.add(TF)

model.add(Flatten())

model.add(Dense(512))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Dense(48)) #activation='softmax')) #mnist사용할 경우
model.add(LeakyReLU(alpha = 0.1))
model.summary()

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
lr = ReduceLROnPlateau( monitor='val_loss', factor=0.3, patience=5, verbose=1, mode='auto')
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])
model.fit(train_dataset, epochs = 100, batch_size = 64, verbose = 1 ,callbacks = [es, lr])

# test data
X_test=[]

for test_path in tqdm(test_paths):
    img=tf.io.read_file(test_path)
    img=tf.image.decode_jpeg(img, channels=3)
    img=tf.image.resize(img, [360,640])
    X_test.append(img)

X_test=tf.stack(X_test, axis=0)
X_test.shape

pred=model.predict(X_test)

submission.iloc[:,1:]=pred

print(submission)
submission.to_csv('C:/data/motion/baseline_submission.csv', index=False)