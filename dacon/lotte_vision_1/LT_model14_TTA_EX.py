Skip to content
Sign up
skanwngud
/
Dacon
00
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Dacon/Lotte/lotte10.py /
@skanwngud
skanwngud commit
Latest commit 10c3f0a 21 hours ago
 History
 1 contributor
133 lines (108 sloc)  3.75 KB
  
# tta 사용

import tqdm
# import numpy as np
# import pandas as pd

# predict = list()
# for i in tqdm(range(15)):

import glob
import numpy as np
import pandas as pd
import os

from PIL import Image

from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from scipy import stats

#0. 변수
filenum = 18
img_size = 128
batch = 16
seed = 42
epochs = 1000

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    width_shift_range= 0.05,
    height_shift_range= 0.05
)


test_dir = 'c:/LPD_competition/test_new/'

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size = (img_size, img_size),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

eff = EfficientNetB4(
    include_top=False,
    input_shape=(128, 128, 3)
)

# 코드 실행시 모든 파일에 000을 붙여준다!
# for i in range(1000):
#     os.mkdir('../data/lpd/train_new/{0:04}'.format(i))

#     for img in range(48):
#         image = Image.open(f'../data/lpd/train/{i}/{img}.jpg')
#         image.save('../data/lpd/train_new/{0:04}/{1:02}.jpg'.format(i, img))

for i in range(72000):
    image = Image.open(f'c:/LPD_competition/test_1/test/{i}.jpg')
    image.save('c:/LPD_competition/test_new/test_new/{0:05}.jpg'.format(i))
    print(str(i) + ' 번째 이미지생성 완료')

# #2. 모델
model = Sequential()
model.add(eff)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='swish'))
model.add(GaussianDropout(0.4))
model.add(Dense(1000, activation='softmax'))

model.load_weights(
    'c:/data/modelcheckpoint/lotte_4.hdf5'
)

sub = pd.read_csv(
    'c:/LPD_competition/sample.csv'
)

save_folder = 'c:/data/csv'

'''
result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - mode')
    pred = model.predict(test_data, steps = len(test_data))
    pred = np.argmax(pred, 1)
    result.append(pred)
    print(f'{tta+1} 번째 제출 파일 저장하는 중')
    temp = np.array(result)
    temp = np.transpose(result)
    temp_mode = stats.mode(temp, axis = 1).mode
    sub.loc[:, 'prediction'] = temp_mode
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)
    temp_count = stats.mode(temp, axis = 1).count
    for i, count in enumerate(temp_count):
        if count < tta/2.:
            print(f'{tta+1} 반복 중 {i} 번째는 횟수가 {count} 로 {(tta+1)/2.} 미만!')
'''

cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data), verbose = 1) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)

    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    count_result.append(count)
    print(f'기록 : {count_result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
