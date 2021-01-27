import numpy as np
import matplotlib.pyplot as plt

#sin함수에 대해 알아보기
x = np.arange(0, 10, 0.1) #arange 0 ~ 10까지 0.1단위로 자르기
y = np.sin(x)

plt.plot(x, y)
plt.show()


