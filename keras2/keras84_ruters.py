'''
1986년에 로이터에서 공개한 짧은 뉴스 기사와 토픽의 집합인 로이터 데이터셋을 사용하겠습니다.
32 이 데이터셋은 텍스트 분류를 위해 널리 사용되는 간단한 데이터셋입니다.
 46개의 토픽이 있으며 어떤 토픽은 다른 것에 비해 데이터가 많습니다. 각 토픽은 
 훈련 세트에 최소한 10개의 샘플을 가지고 있습니다.
'''
from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 10000, test_split = 0.2
)

#print(x_train[0], type(x_train[0]))# <class 'list'>
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 
# 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 
# 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 
# 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
print(y_train[0], type(y_train[0])) #3 scaler 1개 #3 <class 'numpy.int64'> 
print(len(x_train[0]), len(x_train[11])) #87 59
print("===========================")
print(x_train.shape, x_test.shape) #(8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print("뉴스기사 최대길이 : ", max(len(I) for I in x_train)) #뉴스기사 최대길이 :  2376
print("뉴스기사 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #뉴스기사 평균길이 :  145.5398574927633

#그래프
# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

#y_분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y_분포", dict(zip(unique_elements, counts_elements)))
# y_분포 {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 
# 6: 48, 7: 16, 8: 139, 9: 101, 10: 124, 11: 390, 12: 49, 
print("=================================================")
# plt.hist(y_train, bins = 46)
# plt.show()

#x의 단어분포
word_to_index = reuters.get_word_index() #단어들 보여주기
print(word_to_index) #키와 밸류로 구성
print(type(word_to_index)) #'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803.......
print("================================================")


#키와 벨류를 교체
index_to_word = {}
for key, value in word_to_index.items() : 
    index_to_word[value] = key
#10996: 'mdbl', 16260: 'fawc'
print(index_to_word)
print(index_to_word[1]) #the
print(len(index_to_word)) #30979
print(index_to_word[30979]) #northerly

# x_train[0]
print(x_train[0]) #[1, 2, 2, 8, 43, 10, 447, 5, ..
print(' '.join([index_to_word[index] for index in x_train[0]])) # join해서 출력하기
# y 카테고리 갯수 출력
category = np.max(y_train) + 1
print("y 카테고리 개수 : ", category)

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 #24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

#==========================================================================

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(8982,) (2246,) (8982,) (2246,

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 100
x_train = pad_sequences(x_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩 0000
x_test = pad_sequences(x_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩

#원핫코딩
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train) 
# y_test = to_categorical(y_test)

#1. 데이터셋
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 100, shuffle = False)
#print(x_train.shape, x_test.shape, x_val.shape, y_train.shape, y_test.shape, y_val.shape) #(8982,) (2246,) (8982,) (2246,
#(7185,) (2246,) (1797,) (7185,) (2246,) (1797,)

#2. 모델링
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv2D
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim= 64, input_length=100))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20, verbose= 1, mode = 'auto')
rl = ReduceLROnPlateau(monitor='val_loss', patience = 10, factor = 0.3, verbose = 1, mode = 'auto')
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [es, rl])

#4.평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  2.0444114208221436
# acc :  0.6620659232139587

#시각화
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()