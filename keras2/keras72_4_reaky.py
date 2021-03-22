import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-6, 6, 0.01)

def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>=0) * x + (x < 0) * 0.01 * x # 알파값(보통 0.01) 조정가능
    # return np.maximum(0.01*x,x) # same
 
#그래프 출력
plt.plot(x, leakyrelu_func(x), label="Leaky ReLU")
plt.grid()
plt.show()