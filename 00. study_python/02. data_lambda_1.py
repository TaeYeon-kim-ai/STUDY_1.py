def hap(x, y):
  return x + y

hap(10, 20)

#람다 적용
(lambda x,y: x + y)(10, 20)


#==================================
#map(함수, 리스트)
map(lambda x: x ** 2, range(5))             # 파이썬 2
#[0, 1, 4, 9, 16]  
list(map(lambda x: x ** 2, range(5)))     # 파이썬 2 및 파이썬 3
#[0, 1, 4, 9, 16]


#=================================
#reduce(함수, 순서형 자료)
from functools import reduce   # 파이썬 3에서는 써주셔야 해요  
reduce(lambda x, y: x + y, [0, 1, 2, 3, 4])
reduce(lambda x, y: y + x, 'abcde')

#=================================
#filter(함수, 리스트)
filter(lambda x: x < 5, range(10))       # 파이썬 2
#[0, 1, 2, 3, 4]  
list(filter(lambda x: x < 5, range(10))) # 파이썬 2 및 파이썬 3
#[0, 1, 2, 3, 4]

#=================================
#홀수돌리기
filter(lambda x: x % 2, range(10))        # 파이썬 2
#[1, 3, 5, 7, 9]  
list(filter(lambda x: x % 2, range(10)))  # 파이썬 2 및 파이썬 3
#[1, 3, 5, 7, 9]








