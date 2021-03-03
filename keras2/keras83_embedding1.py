#긍정부정 키워드
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '안기가 어색해요',
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
#[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27]]
#- 길이가 일정하지 않아 분석불가
#앞쪽을 0으로 채워야 분석잘됨

#0ㅇ,로 채우기
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', maxlen=4) #maxlen은 앞에걸 자름
#pre
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27]]

#post
# [[ 2  3  0  0]
#  [ 1  4  0  0]
#  [ 1  5  6  7]
print(pad_x)
print(pad_x.shape)
#(13, 5)

print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24
#  25 26 27]  *11없음
print(len(np.unique(pad_x))) #27 *11없음

#2. 모델링
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()

















