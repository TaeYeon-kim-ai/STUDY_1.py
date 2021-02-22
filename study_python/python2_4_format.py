#float자료형 기본
output_a = "{:f}".format(52.273) # 52.273000  실수형 채우기
output_b = "{:15f}".format(52.273) #       52.273000 #앞에 비우고 15칸 맞춰서
output_c = "{:+15f}".format(52.273) #      +52.273000 #앞에 비우고 15칸 맞추고 +추가해서 넣기
output_d = "{:+015f}".format(52.273) # +0000052.273000 #앞에 비우고 15칸 맞추고 0과+ 추가해서 넣기

print(output_a)
print(output_b)
print(output_c)
print(output_d)

