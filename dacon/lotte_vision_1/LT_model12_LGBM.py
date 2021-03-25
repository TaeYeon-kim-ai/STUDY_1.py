#m31로 만든 0.95 이상의 n_component = ?를 사용하여
#XGB 모델을 만들 것

#mnist dnn 보다 성능 좋게 만들어라!!
#cnn과 비교!!
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def get_column_index(model):
    feature = model.feature_importances_
    feature_list = []
    for i in feature:
        feature_list.append(i)
    feature_list.sort(reverse = True)
 
    result = []
    for j in range(len(feature_list)-len(feature_list)//4):
        result.append(feature.tolist().index(feature_list[j]))
    return result

#1. 데이터

x = np.load('../../data/npy/train_x_128.npy', allow_pickle=True)
y = np.load('../../data/npy/train_y_128.npy', allow_pickle=True)
x_pred = np.load('../../data/npy/predict_x_128.npy', allow_pickle=True)
print(x.shape) #(70000, 28, 28)
print(y.shape) 

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])/255.
print(x.shape)

#MinMax
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)

# d = np.argmax(cumsum >= 0.99)+1 #95 가능범위 확인
# #print("cumsum >= 0.99", cumsum >= 0.99)
# print("d : ", d)

# #시각화
# # import matplotlib.pyplot as plt
# # plt.plot(cumsum)
# # plt.grid()
# # plt.show()

# pca = PCA(n_components= d , ) #차원축소
# x2 = pca.fit_transform(x)
# print(x2.shape)
# # d :  331
#(70000, 331)

#1.1 데이터 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 55)
print(x_train.shape, x_test.shape) #(56000, 154) (14000, 154)
print(y_train.shape, y_test.shape) #(56000,) (14000,)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=55)

#Converting the dataset in proper LGB format
import lightgbm as lgb
from lightgbm import LGBMClassifier
d_train=lgb.Dataset(x_train, label = y_train)

# print(x_train.max)
# print(x_train.min)

#2. 모델링
#setting up the parameters
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=10
params['num_class']=3 #no.of unique values in the target class not inclusive of the end value

#training the model
clf=lgb.train(params,d_train,100)  #training the model on 100 epocs

#training the model
clf=lgb.train(params, d_train,100)  #training the model on 100 epocs
#prediction on the test dataset
y_pred_1=clf.predict(x_test)
#printing the predictions
y_pred_1

# y_pred = model.predict(x_test)
# pred_proba = model.predict_proba(x_test)[:1]

# acc = model.score(x_test, y_test)
# print(model.feature_importances_)
# print("acc : ", acc)

# # #시각화
# # import matplotlib.pyplot as plt
# # import numpy as np
# # def plot_feature_importances_dataset(model): 
# #     n_features = dataset.data.shape[1]
# #     plt.barh(np.arange(n_features), model.feature_importances_,
# #         align='center')
# #     plt.yticks(np.arange(n_features), dataset.feature_names)
# #     plt.xlabel("Feature Improtances")
# #     plt.ylabel("Features")
# #     plt.ylim(-1, n_features)

# # plot_feature_importances_dataset(model)
# # plt.show()
