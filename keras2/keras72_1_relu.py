#relu #음수인 부분은 0으로 양수인부분은 그대로 표기
import numpy as np
import matplotlib.pyplot as plt

def relu(x) :
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = relu(x) #여기 넣기 

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()


####과제
# elu, selu, reaky relu
#72_2, 3, 4번으로 파일을 만들 것
