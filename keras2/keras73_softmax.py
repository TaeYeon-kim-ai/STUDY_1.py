import numpy as np
import matplotlib.pyplot as plt

def softmax(x) : 
    return np.exp(x) / np.sum(np.exp(x))


x = np.arange(1, 5)
y = softmax(x)

print(x)
print(y)
#[1 2 3 4]
#[0.0320586  0.08714432 0.23688282 0.64391426]

ratio = y
labels = y
plt.pie(ratio, labels = labels, shadow=True, startangle = 90)
plt.show()