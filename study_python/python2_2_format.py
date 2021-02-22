#foramt() 함수로 숫자를 문자열로 변환
format_a = "{}만원".format(5000)
format_b = "파이썬 열공하여 첫 연봉 {}만원 만들기".format(5000)
format_c = "{}{}{}".format(3000, 4000, 5000)
format_d = "{}{}{}".format(1, "문자열", True)

#출력하기
print(format_a) #5000만원
print(format_b) #파이썬 열공하여 첫 연봉 5000만원 만들기
print(format_c) #300040005000
print(format_d) #1문자열True

'''
format 안의 ,로 구분된 수, 문자열을 "{}"안에 각각 넣어준다.
