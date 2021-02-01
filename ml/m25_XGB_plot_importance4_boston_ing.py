# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor, plot_importance


#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor(max_depth=4)
#model = GradientBoostingRegressor(max_depth=4)
model = XGBRegressor(n_job = -1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

#5. 시각화
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model): 
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Improtances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_importance(model)
plt.show()

#=================================================================


#DecisionTreeRegressor
# [5.74134125e-02 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  7.65831515e-03 2.96399134e-01 3.70931404e-04 5.95459582e-02
#  0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#  5.78612249e-01]
# acc :  0.8774175457631728

#정리후
# [0.63868022 0.22083784 0.09401858 0.02595221 0.02051116 0.
#  0.         0.         0.         0.        ]
# acc col정리 :  0.7701633081053628

#RandomForestRegressor
# [3.63942795e-02 9.13236025e-04 2.43977889e-03 3.17871283e-04
#  1.88725861e-02 4.32529090e-01 8.04460541e-03 5.16515275e-02
#  1.42144762e-03 4.40573811e-03 1.22990566e-02 3.96856950e-03
#  4.26742213e-01]
# acc :  0.9043449042642594

#정리후
# [0.38222511 0.47375412 0.06540497 0.02413377 0.02160692 0.00971394
#  0.00427154 0.00856714 0.00649685 0.00382563]
# acc col정리 :  0.8679631482172111

#model = GradientBoostingRegressor(max_depth=4)
#정리전
# [3.07012563e-02 4.53899150e-04 4.23154396e-03 3.46148856e-04
#  3.17956691e-02 3.47320921e-01 9.98202781e-03 7.83184082e-02
#  4.91739529e-03 1.24129058e-02 2.29679044e-02 1.08648434e-02
#  4.45687077e-01]
# acc :  0.9428522000019474

#정리후
# [0.52018549 0.27342655 0.08721533 0.02310321 0.02308252 0.02069672
#  0.02276342 0.01181393 0.0154727  0.00224015]
# acc col정리 :  0.9022386471564411

#XGBRegressor
#정리전
# [0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
#  0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
#  0.42848358]
# r2 :  0.9221188601856797

#정리후
# [0.39029327 0.25759977 0.05749745 0.08199805 0.06707577 0.04337953
#  0.02392683 0.03295108 0.0299829  0.01529539]
# r2 col정리 :  0.882833592562321