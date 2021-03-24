#TTA((Test Time Augmentation)

# 머신러닝 모델을 학습시킬 때, 
# 데이터가 부족하는 등의 문제가 생기면 데이터 Augmentation을 사용합니다. 
# 기존에 있는 데이터 셋을 회전/반전/뒤집기/늘이기/줄이기/노이즈 등 다양항 방법을 사용하여 부풀립니다.

# Test Time Augmentation(TTA)도 augmentation하는 방법은 같습니다. 하
# 지만 학습할 때 augmentation하는게 아닌, 테스트 셋으로 모델을 테스트하거나, 
# 실제 운영할 때 augmentation을 수행하는 것 입니다.


model = Sequential()
model.add(Conv2D(64,(3,3), activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=10.,
        fill_mode='reflect',
        width_shift_range = 0.1, 
        height_shift_range = 0.1)

train_datagen.fit(x_train)

history = model.fit_generator(train_datagen.flow(x_train, y_train,
                              batch_size=bs),
                              epochs=15,
                              steps_per_epoch=len(x_train)/bs,
                              validation_data=(x_val, y_val))

#acc = 0.7528.
#tf.keras.backend.clear_session

tta_steps = 10

predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(test_datagen.flow(x_val, batch_size=bs, shuffle=False), steps = len(x_val)/bs)
    predictions.append(preds)

final_pred = np.mean(predictions, axis=0)

print(f'Accuracy with TTA: {np.mean(np.equal(np.argmax(y_val, axis=-1), np.argmax(final_pred, axis=-1)))}')

