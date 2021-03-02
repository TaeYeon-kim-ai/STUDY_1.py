####과제
# elu, selu, reaky relu
#72_2, 3, 4번으로 파일을 만들 것
import numpy as np
import matplotlib.pyplot as plt

def selu(x, alp) :
    return (x > 0)*x + (x <= 0) * (alp * (np.exp(x) -1))


x = np.arange(-5, 5, 0.1)
alp = 5

y = selu(x, alp) #여기 넣기 

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

# 8. SELU (Scaled ELU)
# ELU 활성화 함수의 변종
# 완전 연결 층만 쌓아서 신경망을 만들고 모든 은닉층이 SELU 활성화 함수를 사용하면 네트워크가 자기 정규화(self-normalized) 된다고 저자는 주장
# 훈련하는 동안 각 층의 출력이 평균 0과 표준편차 1을 유지하는 경향(그래디언트 소실과 폭주 문제를 막아준다.)
# 다른 활성화 함수보다 뛰어난 성능을 종종 보이지만 자기 정규화가 일어나기 위한 몇가지 조건이 존재
# 1) 입력 특성이 반드시 표준화(평균 0, 표준편차 1)되어야 한다.

# 2) 모든 은닉층의 가중치는 르쿤 정규분포 초기화로 초기화되어야 한다. 케라스에서는 kernel_initializer=’lecun_normal’로 설정

# 3) 네트워크는 일렬로 쌓은 층으로 구성되어야 한다. 순환 신경망이나 스킵 연결과 같은 순차적이지 않은 구조에서 사용하면 자기 정규화되는 것을 보장하지 않는다.