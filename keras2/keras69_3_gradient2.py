import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6 #2차  #미분은 바이어스 1차원 줄임 

#미분
gradient = lambda x : 2*x  -4 #가중치 2   바이어스 -4

x0 = 10
epoch = 30 # 30 = 2
learning_rate = 0.1 #0.1 = 2

print("step\tx\tf(x)") #1템포 건너띄어서 x 1템포 건너띄고 f(x) step x fx
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0))) # 0찍고 1칸띄고 0찍고, xo찍고 f(x0) 찍고 0 10 66

for i in range(epoch) : 
    temp = x0 - learning_rate * gradient(x0) # 10 - 0.1*16 = 8.4
    x0 = temp # 8.4 - 0.1 ........

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0))) # 0찍고 1칸띄고 0찍고, xo찍고 f(x0) 찍고 0 10 66)

plt.plot(epoch, x0, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('x0')
plt.show()


# step    x       f(x)
# 00      10.00000        66.00000
# 01      8.40000 42.96000
# 02      7.12000 28.21440
# 03      6.09600 18.77722
# 04      5.27680 12.73742
# 05      4.62144 8.87195
# 06      4.09715 6.39805
# 07      3.67772 4.81475
# 08      3.34218 3.80144
# 09      3.07374 3.15292
# 10      2.85899 2.73787
# 11      2.68719 2.47224
# 12      2.54976 2.30223
# 13      2.43980 2.19343
# 14      2.35184 2.12379
# 15      2.28147 2.07923
# 16      2.22518 2.05071
# 17      2.18014 2.03245
# 18      2.14412 2.02077
# 19      2.11529 2.01329
# 20      2.09223 2.00851
# 21      2.07379 2.00544
# 22      2.05903 2.00348
# 23      2.04722 2.00223
# 24      2.03778 2.00143
# 25      2.03022 2.00091
# 26      2.02418 2.00058
# 27      2.01934 2.00037
# 28      2.01547 2.00024
# 29      2.01238 2.00015
# 30      2.00990 2.00010    # learning_rate줄어들면서 미분값 2로 수렴   2차 함수 미분값 찾기       
                            #웨이트 2


# x0 = 10, lr = 1, epoch = 30
# step    x       f(x)
# 00      10.00000        66.00000
# 01      -6.00000        66.00000
# 02      10.00000        66.00000
# 03      -6.00000        66.00000
# 04      10.00000        66.00000
# 05      -6.00000        66.00000
# 06      10.00000        66.00000
# 07      -6.00000        66.00000
# 08      10.00000        66.00000
# 09      -6.00000        66.00000
# 10      10.00000        66.00000
# 11      -6.00000        66.00000
# 12      10.00000        66.00000
# 13      -6.00000        66.00000
# 14      10.00000        66.00000
# 15      -6.00000        66.00000
# 16      10.00000        66.00000
# 17      -6.00000        66.00000
# 18      10.00000        66.00000
# 19      -6.00000        66.00000
# 20      10.00000        66.00000
# 21      -6.00000        66.00000
# 22      10.00000        66.00000
# 23      -6.00000        66.00000
# 24      10.00000        66.00000
# 25      -6.00000        66.00000
# 26      10.00000        66.00000
# 27      -6.00000        66.00000
# 28      10.00000        66.00000
# 29      -6.00000        66.00000
# 30      10.00000        66.00000