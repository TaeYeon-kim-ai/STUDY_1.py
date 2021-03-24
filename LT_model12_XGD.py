import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1.DATA
x = np.load('../../data/npy/train_x_160.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_160.npy', allow_pickle=True)
x_pred = np.load('../../data/npy/predict_x_160.npy', allow_pickle=True)

from tensorflow.keras.applications.efficientnet import preprocess_input
x = preprocess_input(x)
x_pred = preprocess_input(x_pred)

#generagtor
idg = ImageDataGenerator(
    zoom_range = 0.3,
    height_shift_range=(-1, 1),
    width_shift_range=(-1, 1),
    rotation_range=40, 
)

idg2 = ImageDataGenerator()

#파라미터튜닝 적용
parameters = [
    {"n_estimators" : [100], "learning_rate " : [0.1, 0.001, 0.5], "max_depth" : [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]
bts = 32

kfold = KFold(n_splits= 5, shuffle = True)
    # print("TRAIN:", train_index, "TEST:", test_index) 
    x_train, x_val = x[train_index], x[test_index] 
    y_train, y_val = y[train_index], y[test_index]

    x_train, y_train = idg.flow(x_train, y_train)
    x_val, y_val = idg2.flow(x_val, y_val)
    test_generator = idg2.flow(x_pred, shuffle=False)

    model = RandomizedSearchCV(XGBClassifier(), parameters, cv = kfold)

    result = model.predict_generator(test_generator,verbose=True)/8

model.summary()

start = datetime.datetime.now()
model.fit(x_train, y_train)
end = datetime.datetime.now()
print("time", end-start)

model.save('C:/data/h5/LT_vision_6.h5')
model.save_weights('C:/data/h5/LT_vision_model2_6.h5')
# model = load_model('C:/data/h5/LT_vision_model2_5_mobileNet.h5')
# model.load_weights('C:/data/h5/LT_vision_5_mobileNet.h5')


score = cross_val_score(model, train_generator, cv= kfold)

print("최적의 매개변수 : ", model.best_estimator_)

acc = model.score(test_generator)
print(model.feature_importances_)
print("acc : ", acc)
