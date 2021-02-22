#문자열의 format() 함수

"{}". format(10)
"{}{}".format(10, 20)
"{}{}{}".format(101, 202, 303, 404, 505)
#앞쪽에 있는 문자열 기호에 format괄호안의 맥변수로대체;


#format()함수로 숫자를 문자열로 변환
string_a = "{}".format(10)

print(string_a)
print(type(string_a))
# 10
# <class 'str'>

#{}기호 양쪽에 다른 문자열을 같이 넣은 형태