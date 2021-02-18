import p11_car 
# 운전하다
# p11_car.py의 module이름은 :  p11_car..... 파일명 불러온 애 __name__은 파일명이 나온다
import p12_tv
# 시청하다
# p11_tv.py의 module이름은 :  p12_tv ..... P12.의 결과와 __name__출력(파일명)

print("============================================")
print("p13_do.py의 module 이름은 : ", __name__) #p13_do.py의 module 이름은 :  __main__
print("============================================")

p11_car.drive()#운전하다
p12_tv.watch()#시청하다

