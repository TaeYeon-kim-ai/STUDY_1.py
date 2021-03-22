x_train = 0.5 
y_train = 0.8 #0.8이 나와야함

#===============================
weight = 0.5 # 직선 #0.3, 0.66, 0.8, 0.9
lr = 0.1 #0.1, 10, 100, 0.001
epoch = 20
#===============================

for iteration in range(epoch) :  #iteration... i라 생각해도 됨
    y_predict = x_train * weight # +bias # 0.25
    error = (y_predict - y_train) **2 #0.3025 #거리만들기(양수) error = loss = cost
    
    print("Error : " + str(error) + "\ty_predict : " + str(y_predict))

    up_y_predict = x_train * (weight + lr) #0.5 * 0.51
    up_error = (y_train - up_y_predict) **2 # 0.8 - (0.5 * 0.51) 거리만들기(양수) 

    down_y_predict = x_train * (weight) # 0.5 * 0.5
    down_error = (y_train - down_y_predict) **2 # 0.8 - (0.5 * 0.5)

    if(down_error <= up_error) : # 0.55 <= 0.545
        weight = weight - lr # 0.49 = 0.5 - 0.1 
    if(down_error > up_error) : # 0.55 > 0.545
        weight = weight + lr # 0.51 = 0.5 + 0.1

'''
epoch = 1
y_predict = 0.5 * 0.5 = 0.25
error = (0.25 - 0.8)**2 = 0.3025
str(error = 0.3025)
str(y_predict = 0.25)
=====
up_y_predict = 0.5 * (0.5 + 0.1) = 0.255
up_error = (0.8 - 0.255)**2 = 0.297025
=====
down_y_predict = 0.5 * 0.5 = 0.25
down_error = (0.8 - 0.25)**2 = 0.3025
=====
if(0.3025 <= 0.297025) :
    0.79 = 0.8 - 0.1

if(0.3025 > 0.297025) : 
    0.81 = 0.8 + 0.1



epoch = 2
y_predict = 0.5 * 0.81 = 0.405


'''



#1바퀴 돌 때 마다 error 갱신, weight갱신됨, 

#weight 0.5, lr 0.1 epoch 200
# Error : 0.0024999999999999935   y_predict : 0.7500000000000001
# Error : 1.232595164407831e-32   y_predict : 0.8000000000000002


