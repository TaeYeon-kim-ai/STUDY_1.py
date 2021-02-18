#class안의 변수는 다 self를 붙인다.

class Person :
    def __init__(self, name, age, address) : #def __init__(self, )<- 필수 
        self.name = name
        self.age = age
        self.address = address

    def greeting(self): #<- 무조건 사용
        print('안녕하세요, 저는 {0}입니다.'.format(self.name)) #0은 뒤에 .format(self.name)과 매치
                            #{0}, {1}    0 1은  뒤에 .format(self.name, aaa)과 매치


# import방식 사용하는 파일이나 하단에 넣어서 import할 수 있고 
# hash .... anaconda에 넣어서 import하고 그건 package라 하고 내부에 하나하나의 요소는 module이라 한다.