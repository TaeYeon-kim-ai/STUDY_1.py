import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from PIL import Image

test_datagen = ImageDataGenerator(rescale=1./255) 

#new_test데이터 정제
new_test = test_datagen.flow_from_directory(
    'C:/data/image/gender_test',
    target_size = (128, 128),
    batch_size = 10,
    class_mode= 'binary',
    save_to_dir='C:/data/image/gender_test_resize'
)

x_train = np.load('C:/data/image/gender_npy/keras67_train_x.npy') #x 1
y_train = np.load('C:/data/image/gender_npy/keras67_train_y.npy')
x_test = np.load('C:/data/image/gender_npy/keras67_test_x.npy') #x 1
y_test = np.load('C:/data/image/gender_npy/keras67_test_y.npy')
new_test = np.load('C:/data/image/gender_npy/keras67_new_test.npy')


#4. 평가
model = load_model('C:/data/h5/male_female_model2.h5')
model.load_weights('C:/data/h5/male_female__weight.h5')

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
results = model.predict(new_test)

# 남/여
if results > 0.5 :
    print("요청한 사진은 ", results*100, "% 확률로 남자입니다.")
else :
    print("요청한 사진은" , (1-results)*100, "% 확률로 여자입니다." )

#요청사진 확인
import matplotlib.pyplot as plt
plt.imshow(new_test[0]) # 1이 남자 0이 여자
plt.show()

# loss :  0.2705157697200775
# acc :  0.9282833933830261
# 요청한 사진은  [[99.23718]] % 확률로 남자입니다.






