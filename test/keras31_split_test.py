import numpy as np

a = np.array(range(1, 11)) # a에 0~10까지 집어 넣는다
size = 5 # 데이터를 5개씩 끊어서 리스트 구성 (5개씩 끊어서) 5개 열로 구성함

def split_x(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1 ): #for반복문 i를 반복해라 size + 1 까지
        subset = seq[i : (i + size)]  #seq를 구성해라 i(1)부터 i+size(5)까지
        aaa.append(subset) # aaa에 추가해라 [] 한바퀴돌
    print(type(aaa)) #aaa 의 타입을 추가해라
    return np.array(aaa) #aaa를 반환하라

dataset = split_x(a, size) #dataset에 추가
print("===========================")
print(dataset) #split_x datasets을 0~10까지 size 5까지 순서대로 넣기


# <class 'list'>
# ===========================
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# def split_x(seq, size): 
#     aaa = []
#     for i in range(len(seq) - size + 1 ):
#         subset = seq[i : (i + size)]  #seq를 구성해라 i(1)부터 i+size(5)까지
#         aaa.append([item for item in subset]) # aaa에 
#     print(type(aaa))
#     return np.array(aaa)