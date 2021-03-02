#sigmoid 0 ~ 1

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x) :
    return 1 / (1 + np.exp(-x)) #activation sigmoid일 때 
                                #레이어의 마지막 값에 return 1 / (1 + np.exp(-x))수식이 추가 됨
x = np.arange(-5, 5, 0.1) #x = -5 ~ 5 까지 0.1  간격으로 표기
y = sigmoid(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()