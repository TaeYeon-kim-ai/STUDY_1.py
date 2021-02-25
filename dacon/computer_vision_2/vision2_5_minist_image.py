from PIL import Image, ImageDraw
from numpy import genfromtxt
import csv
import os

import csv
csv_file = raw_input('C:/data/vision_2/mnist_data/train.csv')
txt_file = raw_input('C:/data/vision_2/mnist_data/train_image_txt')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

'''
bar = '□□□□□□□□□□'	#퍼센트바의 초기 상태
sw = 1
def percent_bar(array,count):   #퍼센트를 표시해주는 함수
    global bar
    global sw
    length = len(array)		
    percent = (count/length)*100	#배열의 길이와 현재 for문의 count의 퍼센트를 구함
    if count == 1 :
        print('preprocessing...txt -> png ')
    print('\r'+bar+'%3s'%str(int(percent))+'%',end='')	#퍼센트바와 퍼센트를 표시
    if sw == 1 :
        if int(percent) % 10 == 0 :
            bar = bar.replace('□','■',1)	#네모가 열개이므로 10퍼센트마다 한개씩 채워줌
            sw = 0
    elif sw == 0 :
        if int(percent) % 10 != 0 :
            sw = 1

def preprocessing_txt(path):
    txtPaths = [os.path.join(path,f) for f in os.listdir(path)]
    count = 0
    #파일읽기
    for txtPath in txtPaths :
        count += 1
        filename = os.path.basename(txtPath)
        percent_bar(txtPaths,count)
        f = open(txtPath)
        img = []
        while True :
            tmp=[]
            text = f.readline()
            if not text :
                break
            for i in range(0,len(text)-1) : 
                #라인을 일어올때 text가 1일경우 255로 변경
                if int(text[i]) == 1 :
                    tmp.append(np.uint8(255))
                else :
                    tmp.append(np.uint8(0))
            img.append(tmp)		#img배열에 쌓아줌
        img = np.array(img)
        cv2.imwrite('./kNN/trainingPNGs/'+filename.split('.')[0]+'.png',img) #저장
    print('\n'+str(count)+'files are saved(preprocessing_txt2png)')
preprocessing_txt('./kNN/trainingDigits')

'''