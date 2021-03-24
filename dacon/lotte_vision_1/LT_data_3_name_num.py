import numpy as np
import cv2 as cv
import glob
from tensorflow.keras.utils import to_categorical

train_image_arr = []
train_y = []
print(train_y)
for i in range(1000):
    path = '../../data/LPD_competition/train/' + str(i) + '/'
    print(path)
    image = cv.imread(path)
    image = cv.resize(image, (64, 64), interpolation = cv.INTER_CUBIC)
    sampleList = random.sample(image, 96)
    train_image_arr.append(sampleList)
    train_y.append(i)

train_image_arr = np.asarray(train_image_arr)
train_y = np.asarray(train_y)
train_y = train_y.reshape((-1, 1))
train_y = to_categorical(train_y)

print(train_image_arr.shape)
print(train_y)
print(train_y.shape)

np.save('../../data/npy/train_x_64.npy', arr = train_image_arr)
np.save('../../data/npy/train_y_64.npy', arr = train_y)

train_fnames = natsorted(os.listdir(TRAIN_DIR)) # 1000
test_fnames = natsorted(os.listdir(TEST_DIR))
i = 0

for idx, folder in enumerate(train_fnames): # 트레인 폴더는 1000개 있다
    # if idx >= 1:
    #     break
    print("folder :", folder)
    base_dir = TRAIN_DIR + '/' + folder + '/' # 각 폴더 디렉토리 'C:/data/LPD_competition/train/0'
    print("basedir :", base_dir)
    img_lst = natsorted(os.listdir(base_dir)) # 각 폴더 안에 있는 jpg 파일 리스트)
    
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(i) + '.jpg'
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        i += 1

###############################################################

test_image_arr = []
for i in range(72000):
    path = '../../data/LPD_competition/test/' + str(i) + '.jpg'
    image = cv.imread(path)
    image = cv.resize(image, (64, 64), interpolation = cv.INTER_CUBIC)
    test_image_arr.append(image)
    print(i)
test_image_arr = np.asarray(test_image_arr)

np.save('../../data/npy/predict_x_64.npy', arr = test_image_arr)