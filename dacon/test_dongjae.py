import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)

x = [] #x에 첨부
for i in range(1,2): # i  1~2까지 소스 가져오기
    df = pd.read_csv(f'C:\STUDY\dacon\data\submission_{i}.csv', index_col=0, header=0) # 데이터 프레임 가져오기
    data = df.to_numpy() # df를  numpy로 전환
    x.append(data) 

x = np.array(x)

df = pd.read_csv(f'C:\STUDY\dacon\data\submission_{i}.csv', index_col=0, header=0)
for i in range(7776):
    for j in range(9):
        a = []
        for k in range(1):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('C:\STUDY\dacon\data\sample_submission_check3.csv')  


