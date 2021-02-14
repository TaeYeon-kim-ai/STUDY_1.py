import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, AveragePooling2D, AvgPool2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


#train/test준비 선언
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255) 

#xy_train
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128), 
    batch_size = 32, 
    class_mode = 'binary',
    subset="training"
)

xy_val = train_datagen.flow_from_directory(
    'C:/data/image/gender_generator',
    target_size = (128, 128), 
    batch_size = 32, 
    class_mode = 'binary',
    subset="validation"
)

#xy로 한 이유는 .flow_from_directory통과하면 x data와 y data가 생성됨
xy_test = test_datagen.flow_from_directory(
    'C:/data/image/gender',
    target_size = (128, 128),
    batch_size = 32, 
    class_mode = 'binary'
)

xy1_test = test_datagen.flow_from_directory(
    'C:/data/image/gender_test',
    target_size = (128, 128),
    batch_size = 32, 
    class_mode = 'binary'
)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D
input1 = Input(shape=(128, 128 ,3))
x = Conv2D(64, 4, activation='relu')(input1)
x = BatchNormalization()(x)
x = Conv2D(128, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 2, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(256, 2, activation='relu')(x)
x = AveragePooling2D()(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Dropout(0.4)(x)
x = Flatten()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation= 'sigmoid')(x)
model = Model(inputs = input1, outputs = outputs)
model.summary()

model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(learning_rate=0.1), 
    metrics=['acc']
    
    )

es=EarlyStopping(patience=10, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=5, verbose=1, monitor='loss')

hist= model.fit_generator(
    xy_train,
    steps_per_epoch=44,
    epochs=50,
    validation_data=xy_val,
    callbacks=[es, rl]
)
#평가
loss, acc = model.evaluate(xy_test)
print('loss : ', loss)
print('acc : ', acc)

#남/여
results = model.predict(xy1_test)
if results > 0.5 :
    print("남자일 확률 : " , results*100, "%")
else :
    print("여자일 확률 : " , (1-results)*100, "%" )


#시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#그래프 출력
import matplotlib.pyplot as plt
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
