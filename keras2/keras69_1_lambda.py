#lambda 함수

#람다로 정의
gradient = lambda x: 2*x - 4  #가중치 2, 바이어스 -4

#일반 함수로 정의
def gradient2(x) :
    temp = 2*x - 4
    return temp

x = 3

print(gradient(x))
print(gradient2(x))
# 2
# 2

