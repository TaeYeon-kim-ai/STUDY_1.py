# 컬럼 drop

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import r2_score

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size =0.8, random_state = 66, shuffle=True)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor(max_depth=4)
# model = GradientBoostingRegressor(max_depth=4)
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

#=================================================================

# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)


#DecisionTreeRegressor
# [0.03400704 0.         0.26623557 0.11279298 0.         0.
#  0.01272153 0.00124721 0.51986371 0.05313196]
# acc :  0.33339660919782466

#정리후
# [0.51214594 0.33393422 0.0856219  0.01202884 0.01669709 0.03530176
#  0.         0.00427026]
# acc col정리 :  0.38349331181632773

#RandomForestRegressor
# [0.03249957 0.00236945 0.32819722 0.10892196 0.01444619 0.02504835
#  0.01715777 0.01222231 0.4067378  0.0523994 ]
# acc :  0.41502320521898084

#정리후
# [0.38222511 0.47375412 0.06540497 0.02413377 0.02160692 0.00971394
#  0.00427154 0.00856714 0.00649685 0.00382563]
# acc col정리 :  0.8679631482172111

#model1 = GradientBoostingRegressor(max_depth=4)
#정리전
# [0.07378113 0.01309454 0.22903958 0.11018153 0.03224211 0.06459372
#  0.0323457  0.02632961 0.35591914 0.06247293]
# acc :  0.35373091823017033

#정리후
# [0.39319594 0.22452784 0.10333879 0.05565111 0.06917675 0.05265098
#  0.05238476 0.04907383]
# acc col정리 :  0.46617038120420096

#XGBRegressor
#정리전
# [0.02593722 0.03821947 0.19681752 0.06321313 0.04788675 0.05547737
#  0.07382318 0.03284872 0.3997987  0.06597802]
# r2 :  0.23802704693460175

#정리후
# [0.21371435 0.23727222 0.07780732 0.08199577 0.12039103 0.06126727
#  0.0556867  0.1518653 ]
# r2 col정리 :  0.3480727891889245
