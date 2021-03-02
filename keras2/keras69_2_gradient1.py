import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100) #-1 ~ 6까지 100개를(동일간격) x에 넣기
y = f(x)

#그림
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.xlabel('y')
plt.show()
#2차함수 그래프

