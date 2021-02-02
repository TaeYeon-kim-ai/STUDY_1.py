# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier,plot_importance

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
#model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_job = -1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc : ", acc)

#5. 시각화
import matplotlib.pyplot as plt
import numpy as np

'''
def plot_feature_importances_dataset(model): 
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Improtances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
'''
plot_importance(model)
plt.show()


#DecisionTreeClassifier
# [0.         0.0624678  0.         0.         0.         0.
#  0.         0.         0.         0.         0.01297421 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.02364429 0.         0.01695087 0.         0.75156772
#  0.00738884 0.00492589 0.00485651 0.11522388 0.         0.        ]
# acc :  0.9385964912280702

#정리후
# [0.74531618 0.14842193 0.04853149 0.         0.04352429 0.00323162
#  0.00858948 0.         0.00238501 0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.        ]
# acc col정리 :  0.9385964912280702

#RandomForestClassifier
# 정리전
# [0.04519016 0.01122636 0.02751801 0.06340525 0.00342164 0.01154781
#  0.05388663 0.1415684  0.0014222  0.00276204 0.00926887 0.00256837
#  0.01094562 0.05619956 0.00266807 0.00397942 0.00353496 0.00471502
#  0.00214234 0.00396321 0.0962297  0.02295978 0.13267572 0.10970825
#  0.0103873  0.00754772 0.04389286 0.09998166 0.00923926 0.00544381]
# acc :  0.9649122807017544

#정리후
# [0.08409454 0.13123727 0.14571247 0.12267746 0.11149456 0.05616319
#  0.05052071 0.05434235 0.03403112 0.01847065 0.08910373 0.01823553
#  0.00201157 0.0170707  0.01030759 0.00904531 0.00473572 0.00891811
#  0.01327905 0.00511945 0.00695334 0.00330511 0.00317045]
# acc col정리 :  0.9385964912280702

#GradientBoostingClassifier
#정리전
# [2.68768337e-05 6.05859874e-02 9.85792370e-04 2.58723784e-03
#  5.32392604e-03 2.70448173e-04 1.54224195e-03 7.61611698e-02
#  4.90669525e-04 3.71977555e-04 6.43581371e-03 2.29574609e-05
#  2.73080551e-03 1.10345124e-02 2.65651460e-03 6.58010941e-03
#  2.47358031e-03 8.29599538e-04 4.92183387e-04 2.43477756e-03
#  4.84372127e-02 2.71373090e-02 2.34103491e-03 6.14494518e-01
#  1.33797935e-03 5.80635683e-04 9.27163702e-03 1.10287750e-01
#  1.42958634e-03 6.45156372e-04]
# acc :  0.956140350877193

#정리후
# [2.29528628e-02 6.06564358e-02 7.55499216e-02 3.89390602e-02
#  5.65698668e-02 3.81017746e-02 7.48302220e-03 3.34414800e-03
#  1.25351050e-03 4.08393014e-03 1.18120333e-03 3.52478068e-03
#  4.97840012e-04 1.83781537e-03 7.18710770e-04 4.48620249e-04
#  6.50689261e-01 1.87092711e-04 5.24356166e-03 3.29582422e-03
#  3.16127559e-05 2.08139819e-02 2.59516283e-03]
# acc col정리 :  0.9298245614035088

#XGBClassifier
#정리전
# [0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
#  0.0054994  0.09745205 0.00340272 0.00369179 0.00769184 0.00281184
#  0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
#  0.00639412 0.0050556  0.01813928 0.02285903 0.22248562 0.28493083
#  0.00233393 0.         0.00903706 0.11586285 0.00278498 0.00775311]
# acc :  0.9736842105263158

#정리후
# [0.09107275 0.5519179  0.05745017 0.11336999 0.00199268 0.04505013
#  0.00300741 0.01859892 0.04313105 0.         0.00667541 0.00213202
#  0.01510823 0.00143737 0.00533567 0.00379835 0.00514523 0.01878964
#  0.00308721 0.0033394  0.00253823 0.00702225 0.        ]
# acc col정리 :  0.956140350877193