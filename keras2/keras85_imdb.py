#과제 = 감정분석 embedding
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters, imdb


(x_train, y_train), (x_test, y_test) = imdb.load_data(
num_words = 10000)

print(x_train[0], type(x_train[0]))
print(x_train[0])
print(len(x_train[0]), len(x_train[11]))
print("===================================")
print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)

print("데이터셋 최대길이 ", max(len(i) for i in x_train)) #데이터셋 최대길이  2494
print("데이터셋 평균길이 ", sum(map(len, x_train)) / len(x_train)) #데이터셋 평균길이  238.71364

# len_result = [len(s) for s in x_train]
# #그래프
# plt.subplot(1,2,1)
# plt.boxplot(len_result)
# plt.subplot(1,2,2)
# plt.hist(len_result, bins=50)
# plt.show()

#y_분포
unique_elements, counts_elements = np.unique(y_train, return_counts = True)
print("y_분포", dict(zip(unique_elements, counts_elements)))
 #y_분포 {0: 12500, 1: 12500}

#x_단어분포
word_to_index = imdb.get_word_index()
print(word_to_index)
print(type(word_to_index))

#키와 밸류 교체
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
# print(index_to_word)
# print(index_to_word[1])
print(len(index_to_word)) #88584
print(index_to_word[88584]) #1

for index, token in enumerate(("<pad>", "<sos>", "<unk>")) :
    index_to_word[index] = token

#print(' '.join([index_to_word[index] for index in x_train[0]]))

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 100
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 256, shuffle = True)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape) #(20000, 100) (5000, 100) (20000,) (5000,)

x_train = x_train.reshape(-1, 100, 1)
x_test = x_test.reshape(-1, 100, 1)

#모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv2D, Flatten, BatchNormalization, Dropout
model = Sequential()
#model.add(Embedding(input_dim = 10000, output_dim = 230, input_length = 100))
model.add(LSTM(128, input_shape = (100, 1)))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 30, verbose= 1, mode = 'auto')
rl = ReduceLROnPlateau(monitor='val_loss', patience = 20, factor = 0.3, verbose = 1, mode = 'auto')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 200, batch_size = 128, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [es, rl])


#4.평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  1.9248653650283813
# acc :  0.8248800039291382

#시각화
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()