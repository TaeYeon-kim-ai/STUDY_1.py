####과제
# elu, selu, reaky relu
#72_2, 3, 4번으로 파일을 만들 것
import numpy as np
import matplotlib.pyplot as plt

def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)
 
#그래프 출력
plt.plot(x, elu_func(x), linestyle='--', label="ELU(Exponential linear unit)")

x = np.arange(-6, 6, 0.01)
y = elu(x) #여기 넣기 

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()
 
 
#######################################################
# 선형함수들
#######################################################
def identity_func(x): # 항등함수
    return x
 
#그래프 출력
plt.plot(x, identity_func(x), linestyle='--', label="identity")
 
 
def linear_func(x): # 1차함수
    return 1.5 * x + 1 # a기울기(1.5), Y절편b(1) 조정가능
 
#그래프 출력
plt.plot(x, linear_func(x), linestyle='--', label="linear")
 
 
#######################################################
# 계단함수들
#######################################################
def binarystep_func(x): # 계단함수
    return (x>=0)*1
    # return np.array(x>=0, dtype = np.int) # same result
 
    # y = x >= 0
    # return y.astype(np.int) # Copy of the array, cast to a specified type.
    # same result
 
#그래프 출력
plt.plot(x, binarystep_func(x), linestyle='--', label="binary step")
 
 
def sgn_func(x): # 부호함수(sign function)
    return (x>=0)*1 + (x<=0)*-1
 
#그래프 출력
plt.plot(x, sgn_func(x), linestyle='--', label="sign function")
 
 
#######################################################
# Sigmoid계열
#######################################################
 
def softstep_func(x): # Soft step (= Logistic), 시그모이드(Sigmoid, S자모양) 대표적인 함수
    return 1 / (1 + np.exp(-x))
 
#그래프 출력
plt.plot(x, softstep_func(x), linestyle='--', label="Soft step (= Logistic)")
 
def tanh_func(x): # TanH 함수
    return np.tanh(x)
    # return 2 / (1 + np.exp(-2*x)) - 1 # same
 
#그래프 출력
plt.plot(x, tanh_func(x), linestyle='--', label="TanH")
 
 
def arctan_func(x): # ArcTan 함수
    return np.arctan(x)
 
#그래프 출력
plt.plot(x, arctan_func(x), linestyle='--', label="ArcTan")
 
 
def softsign_func(x): # Softsign 함수
    return x / ( 1+ np.abs(x) )
 
#그래프 출력
plt.plot(x, softsign_func(x), linestyle='--', label="Softsign")
 
 
 
#######################################################
# ReLU계열
#######################################################
 
def relu_func(x): # ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>0)*x
    # return np.maximum(0,x) # same
 
#그래프 출력
plt.plot(x, relu_func(x), linestyle='--', label="ReLU")
 
 
def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>=0)*x + (x<0)*0.01*x # 알파값(보통 0.01) 조정가능
    # return np.maximum(0.01*x,x) # same
 
#그래프 출력
plt.plot(x, leakyrelu_func(x), linestyle='--', label="Leaky ReLU")
 
 
def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)
 
#그래프 출력
plt.plot(x, elu_func(x), linestyle='--', label="ELU(Exponential linear unit)")
 
 
def trelu_func(x): # Thresholded ReLU
    return (x>1)*x # 임계값(1) 조정 가능
 
#그래프 출력
plt.plot(x, trelu_func(x), linestyle='--', label="Thresholded ReLU")
 
 
 
#######################################################
# 기타계열
#######################################################
 
def softplus_func(x): # SoftPlus 함수
    return np.log( 1 + np.exp(x) )
 
#그래프 출력
plt.plot(x, softplus_func(x), linestyle='--', label="SoftPlus")
 
 
def bentidentity_func(x): # Bent identity
    return (np.sqrt(x*x+1)-1)/2+x
 
#그래프 출력
plt.plot(x, bentidentity_func(x), linestyle='--', label="Bent identity")
 
 
def gaussian_func(x): # Gaussian
    return np.exp(-x*x)
 
#그래프 출력
plt.plot(x, gaussian_func(x), linestyle='--', label="Gaussian")
 
#plt.plot(x, y_identity, 'r--', x, relu_func(x), 'b--', x, softstep_func(x), 'g--')
plt.ylim(-5, 5)
plt.legend()
plt.show()
 
 
Colored by Color Scripter