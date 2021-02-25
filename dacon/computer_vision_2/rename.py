import os 
# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
#경로 지정하고
# os.listdir에 밀어 넣기
# i = 1로 파일이름 지정(1부터 시작)
# dst str(i) + '.jpg'로 파일명 정의
#os.rename(src, dst) 메서드는 파일 또는 디렉토리(폴더) src의 이름을 dst로 변경.

#train_nomal
file_path = 'C:/data/vision_2/mnist_data/train_image/noise' 
file_names = os.listdir(file_path)
file_names

i = 1 #1부터 변경할 것임
for name in file_names: 
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1

#train_ad
file_path = 'C:/data/fish_data/train2_fish_illness' 
file_names = os.listdir(file_path)
file_names

i = 1 #1부터 변경할 것임
for name in file_names: 
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
