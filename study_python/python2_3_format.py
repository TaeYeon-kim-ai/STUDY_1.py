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





