#문자열, 함수(매개변수, 매개변수)
#주어.동사(목적어, 목적어)

#index Error 예외처리
"{}{}".format(1,2,3,4,5)


#"{}{}{}".format(1,2)

'''
{}{}{}보다 format길이가 긴 것은 괜찮으나, 짧으면 range에러 출력

'''

#정수를 특정 칸에 출력
output_a = "{:d}".format(52)

#특정 칸에 출력하기
output_b = "{:5d}".format(52) #5칸
output_c = "{:10d}".format(52) #10칸

#빈칸을 0으로 채우기
output_d = "{:05d}".format(52)
output_e = "{:05d}".format(-52)

print(output_a) # 52
print(output_b) #    52
print(output_c) #         52
print(output_d) # 00052
print(output_e) # -0052

#조합하기
output_h = "{:+5d}".format(52) #   +52 기호를 뒤로 밀기 / 양수
output_i = "{:+5d}".format(-52) #   -52 기호를 뒤로밀기 / 음수
output_j = "{:=+5d}".format(52) # +  52 / 기호를 앞으로 밀기 / 양수
output_k = "{:=+5d}".format(-52) # -  52 / 기호를 앞으로 밀기 /음수
output_l = "{:+05d}".format(52) # +0052 /기호 띄운곳 내 0으로 채우기 / 양수
output_m = "{:+05d}".format(-52) # -0052 / 기호 띄운곳 내 0으로 채우기 / 음수

print("조합하기")
print(output_h)
print(output_i)
print(output_j)
print(output_k)
print(output_l)
print(output_m)







