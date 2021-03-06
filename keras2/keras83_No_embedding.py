
# 임베딩 하는 이유
# 단어사전이 10000개라 해도 원핫인코딩하면 의미가 없음, 그래서 원핫인코딩 하지말고 임베딩 데이터를 통해서 벡터화하면 왕과 여왕 등 수치로 조정하고 
# 10만개의 단어를 원핫인코딩 하면 길어지고 압축성이 없고 유사성과 방향성을 나타내기 힘드니까 데이터를 수치화 한 다음 인베딩을 통과시키겠다.
# 임베딩 레이어를 통과시킬 때 유니크 한 단어를 추출해 숫자로 반환, ex 28개
# 유니크한 단어 숫자 이상으로 input할 경우 연산 숫자는 늘어나나, 오류는 나지 않음 max치까지만 가져감.
# embedding (input_dim = 28, output_dim= 11, input_length=5) 에선 flatten이 먹히나, 자동으로 length값을 받아오는 embedding(28, 11)에는 안먹힌다
# 오류내용 : 
# model.add(Conv1D(32,2))를 사용할 경우 Flatten을 안해줘도 Dense모델에 먹히긴 하나, 뒤에 Flatten해주고 Dense로 전환해주는게 안정적이다.
# *Dense모델은 3차원도 받아들리니까.



#긍정부정 키워드
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘생기긴 했어요',
       ]

# 긍정1, 부정0

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다
# ': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미': 21, '없어요': 22, '재미없다': 23, '재밋네요': 24, '규현이가': 25, '잘생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)

#01,로 채우기
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', maxlen=5) #maxlen은 앞에걸 자름

print(pad_x)
print(pad_x.shape)
#(13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x))) #28 *11없음 0패딩까지 28개
pad_x = pad_x.reshape(13, 5, 1)
print(pad_x.shape)

#2. 모델링
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Embedding, Dense, LSTM, Flatten, Conv1D

#[실습] embedding layer빼고 모델구성
model = Sequential()
model.add(LSTM(32, input_shape = (pad_x.shape[1], pad_x.shape[2])))
model.add(Flatten()) # 안해줘도 먹히긴 하나, 정상적이진 않아요
model.add(Dense(32, activation='relu')) # output 2차원
model.add(Dense(1, activation='sigmoid')) # output 2차원
model.summary()

#2. 컴파일, 훈련
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs = 100)

acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)

#acc :  0.9230769276618958