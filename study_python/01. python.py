print("hello Coding python")

#표현식
#문장 + 문장 = 프로그램

#파이썬 키워드 확인
import keyword 
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 
# 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
# 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 
# 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

print("하나만 출력함")
print("Hello Python Programming...!")

#1. 문자 거꾸로 쓰기
print("문자를 뒤에서부터 선택해볼까요")
print("안녕하세요"[-1])
print("안녕하세요"[-2])
print("안녕하세요"[-3])
print("안녕하세요"[-4])
print("안녕하세요"[-5])

#2. 슬라이싱
print("안녕하세요"[1:4])
#녕하세
print("안녕하세요"[0:2])
#안녕
print("안녕하세요"[1:3])
#녕하
print("안녕하세요"[2:4])
#하세

#리스트 내 슬라이싱 시  [0부터 시작, -1까지 잘림]

#문자열의 범위 선택 연산자.
#뒤의 값을 생략할 때  자동으로 가장 최대위치, 마지막 글자 까지
#앞의 값을 생략할 때 가장 앞쪽의 위치(첫번째 글자 까지 지정)
print("안녕하세요"[1:])
#녕하세요
print("안녕하세요"[:3])
#안녕하

#3. 문자열 길이 구하기
#len() 함수 : 문자열의 수를 구해줌
print(len("인녕하세요"))
#5

#4. 숫자
#숫자의 종류
#a. 자료형 확인하기
print(type(52)) #<class 'int'>
print(type(52.273)) #<class 'float'>

#5. 문자열 연산자
#나머지 연산자 %
#제곱연산자 **

#TypeError 예외
string = "문자열"
number = 273
# string + number
# File "c:\STUDY\00. study_python\python 2021.01.26py", line 63, in <module>
# string + number
# TypeError: can only concatenate str (not "int") to str

#문자열 우선순위
print("안녕" + "하세요" *3)
#안녕하세요하세요하세요
print(("안녕"+"하세요")*3)
#안녕하세요안녕하세요안녕하세요
print("안녕" + ("하세요"*3))
#안녕하세요하세요하세요
print("안녕" + "하세요"*3)
#안녕하세요하세요하세요
print("안녕"+ ("하세요" *3))
#안녕하세요하세요하세요


#2-3 변수와 입력
#변수를 저장할 떄 사용하는 식별자
#활용법
#1. 변수를 선언
#2. 변수에 값을 할당하는 방법
#3. 변수를 참조하는 방법

pi = 3.14
print(pi)
#3.14

#2-3 원의 둘레와 넓이 구하기
#변수선언, 할당
pi = 3.14159265
r = 10

#변수 참조
print("원주율 : ", pi)
print("반지름 : ",  r)
print("원의둘레 : ", 2*pi*r)
print("원의넓이 : ", pi*r*r)
# 원주율 :  3.14159265
# 반지름 :  10
# 원의둘레 :  62.831853
# 원의넓이 :  314.159265

#복합대입연산자
#+- 숫자 덧셈 후 대입
#-= 숫자 뺼셈 후 대입
#*= 숫자 곱셈 후 대입......
string = "안녕하세요"
string += "안녕하세요"
string += "!"
string += "!"
print("string : ", string)
#string :  안녕하세요안녕하세요!!


# #사용자 입력 : input
# string = input("인사말을 입력하세요>")
# print(string)

# #입력자료형 확인
# #입력받기
# string = input("입력>")

# #출력하기
# print("자료:", string)
# print("자료형", type(string))

# #입력받기
# string = input("입력> ")

# #출력
# print("입력 + 100", string + 100)


#문자열 숫자로 바꾸기
# int() #문자열을  int함수로 반환, int는 정수임
# float() #문자열을 float자료형으로 반환, float는 실수임

#int함수 활용하기
# string_a = input("입력A>")
# int_a = int(string_a)

# string_b = input("입력B>")

# int_b = int(string_b)

# print("문자열 자료:", string_a + string_b)

# print("숫자자료:", int_a + int_b)

#int함수와 float함수 활용하기
output_a = int("52")
output_b = float("52.273")

print(type(output_a), output_a) #<class 'int'> 52
print(type(output_b), output_b) #<class 'float'> 52.273

#int함수와 float함수 조립하기
input_a = float(input("첫 번째 숫자> "))
input_b = float(input("두 번쨰 숫자> "))

print("덧셈 결과:", input_a + input_b)
#덧셈 결과: 3.0
print("뺄셈 결과:", input_a - input_b)
#뺄셈 결과: -1.0
print("곱셈 결과:", input_a * input_b)
#곱셈 결과: 2.0
print("나눗셈 결과:", input_a / input_b)
#나눗셈 결과: 0.5

#숫자를 문자열로 변환
#str(다른자료형)
output_a = str(52)
output_b = str(52.273)

print(type(output_a), output_a)
print(type(output_b), output_b)


#2-4 숫자와 문자열의 다양한 기능
















































