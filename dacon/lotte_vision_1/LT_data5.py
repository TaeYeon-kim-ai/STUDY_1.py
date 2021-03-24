import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    rescale=1./255,
    rotation_range=48,          # 지정된 각도 범위내에서 임의로 원본이미지를 회전
    zoom_range=0.2,             # 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소
    height_shift_range=(-1, 1), # 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동
    width_shift_range=(-1, 1),  # 지정된 수평방향 이동 범위내에서 임의로 원본이미지를 이동
    shear_range=0.2,            # 밀림 강도 범위내에서 임의로 원본이미지를 변형
)

# train_generator
train_data = train_datagen.flow_from_directory(
    '../../data/LPD_competition/train', 
    target_size=(160, 160),
    batch_size=48000,                              # 출력되는 y값 개수 설정
    class_mode='sparse'
)

print(train_data[0][0].shape)  # (48000, 64, 64, 3) 
print(train_data[0][1].shape)  # (48000, 1000)
# print(xy_test[0][0].shape)  # (9000, 64, 64, 3)
# print(xy_test[0][1].shape)  # (9000, 1000) 

np.save('../study/LPD_COMPETITION/npy/x_data.npy', arr=train_data[0][0])
np.save('../study/LPD_COMPETITION/npy/y_data.npy', arr=train_data[0][1])

x = np.load('../study/LPD_COMPETITION/npy/x_data.npy')
y = np.load('../study/LPD_COMPETITION/npy/y_data.npy')

print(x.shape, y.shape)