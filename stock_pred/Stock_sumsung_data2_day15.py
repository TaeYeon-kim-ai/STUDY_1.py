
import numpy as np
import pandas as pd

df = pd.read_csv("../data/csv/삼성전자.csv", index_col=0, header=0, encoding='CP949')
df1 = pd.read_csv("../data/csv/삼성전자2.csv", index_col=0, header=0, encoding='CP949')
# print(df.shape)
# print(df.info())
# print(df.columns)
print(df1.shape)
print(df1.info())
print(df1.columns)

#df1 전처리
df = df.drop(df.index[0])
print(df.shape)
print(df1.shape)

df1 = df1.drop(['전일비','Unnamed: 6'], axis=1)
print(df1)
df1 = df1.iloc[[0,1]]
df = pd.concat([df1, df])
print(df) #[2401 rows x 14 columns]

#Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관
# ','외인(수량)', '외국계', '프로그램', '외인비'],dtype='object')

#1. 데이터 전처리
#1.1 특수 기호제거           ^(공백), W(알파벳), S(숫자), repl=r'_'(언더바)
df['시가'] = df['시가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['고가'] = df['고가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['저가'] = df['저가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['종가'] = df['종가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
#df['등락률'] = df['등락률'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['거래량'] = df['거래량'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['금액(백만)'] = df['금액(백만)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
#df['신용비'] = df['신용비'].str.replace(pat=r'[^\w]', repl=r'', regex=True) 
df['개인'] = df['개인'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['기관'] = df['개인'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['외인(수량)'] = df['외인(수량)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['외국계'] = df['외국계'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['프로그램'] = df['프로그램'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
#df['외인비'] = df['외인비'].str.replace(pat=r'[^\w]', repl=r'', regex=True)

# 문자형 숫자로 변환
for col in df.columns : 
    df[col] = pd.to_numeric(df[col])
print(df.info())
print(df)

#1. 일자를 기준으로 오름차순
df_sorted = df.sort_values(by='일자', ascending=True)
print(df_sorted)

# 2. 예측하고자 하는 값을 뒤에 추가
y = df_sorted.iloc[:,3:4]
del df_sorted['종가']
df_sorted['종가'] = y
print(df_sorted)
print(df_sorted.columns)


# 3. 결측값이 들어있는 행 전체 제거
print (df_sorted.isnull().sum()) #거래량, 금액 각 3 18.04.30 ~ 18.05.02 3일
df_dop_null = df_sorted.dropna(axis=0)
print(df_dop_null.shape) #(2397, 14)
# print(df_dop_null)

# 4. 가격 조정(시가, 고가, 저가, 종가, 거래량, 금액, 개인, 기관, 외인, 외국계, 프로그램)
# 시가
a = df_dop_null.iloc[:1735,0] / 50
b = df_dop_null.iloc[1735:,0]
df_dop_null['시가'] = pd.concat([a,b])
print(df_dop_null['시가'])
print(df_dop_null['시가'].shape)

# 고가
a = df_dop_null.iloc[:1735,1] / 50
b = df_dop_null.iloc[1735:,1] 
df_dop_null['고가'] = pd.concat([a,b])
print(df_dop_null['고가'])
print(df_dop_null['고가'].shape)

# 저가
a = df_dop_null.iloc[:1735,2] / 50
b = df_dop_null.iloc[1735:,2]
df_dop_null['저가'] = pd.concat([a,b])
print(df_dop_null['저가'])
print(df_dop_null['저가'].shape)

# 거래량
a = df_dop_null.iloc[:1735,4] *50
b = df_dop_null.iloc[1735:,4]
df_dop_null['거래량'] = pd.concat([a,b])
print(df_dop_null['거래량'])
print(df_dop_null['거래량'].shape)

# 금액(백만)
a = df_dop_null.iloc[:1735,5] *50
b = df_dop_null.iloc[1735:,5]
df_dop_null['금액(백만)'] = pd.concat([a,b])
print(df_dop_null['금액(백만)'])
print(df_dop_null['금액(백만)'].shape)

# 개인
a = df_dop_null.iloc[:1735,7] *50
b = df_dop_null.iloc[1735:,7]
df_dop_null['개인'] = pd.concat([a,b])
print(df_dop_null['개인'])
print(df_dop_null['개인'].shape)

# 기관
a = df_dop_null.iloc[:1735,8] *50
b = df_dop_null.iloc[1735:,8]
df_dop_null['기관'] = pd.concat([a,b])
print(df_dop_null['기관'])
print(df_dop_null['기관'].shape)

# 외인(수량)
a = df_dop_null.iloc[:1735,9] *50
b = df_dop_null.iloc[1735:,9]
df_dop_null['외인(수량)'] = pd.concat([a,b])
print(df_dop_null['외인(수량)'])
print(df_dop_null['외인(수량)'].shape)

# 외국계
a = df_dop_null.iloc[:1735,10] *50
b = df_dop_null.iloc[1735:,10]
df_dop_null['외국계'] = pd.concat([a,b])
print(df_dop_null['외국계'])
print(df_dop_null['외국계'].shape)

# 프로그램
a = df_dop_null.iloc[:1735,11] *50
b = df_dop_null.iloc[1735:,11]
df_dop_null['프로그램'] = pd.concat([a,b])
print(df_dop_null['프로그램'])
print(df_dop_null['프로그램'].shape)

# 종가
a = df_dop_null.iloc[:1735,13] / 50
b = df_dop_null.iloc[1735:,13]
df_dop_null['종가'] = pd.concat([a,b])
print(df_dop_null['종가'])
print(df_dop_null['종가'].shape)
print(df_dop_null)

#5. 상관계수 확인

# print(df.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rc('font', family='Malgun Gothic')
# sns.set(font_scale=1.2)#폰트크기
# sns.heatmap(data=df_dop_null.corr(), square=True, annot=True, cbar=True)
# plt.show()

#6. 열제거(분석하고자 하는것 남기기)
#시가, 고가, 저가, 종가, 거래량, 금액(백만), 개인, 기관, 외인, 외국계, 프로그램
# (['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관','외인(수량)', '외국계', '프로그램', '외인비'],dtype='object')

del df_dop_null['등락률']
del df_dop_null['신용비']
del df_dop_null['프로그램']


print(df_dop_null)

#7. 최종 데이터 확인
print(df_dop_null.shape) #[2397 rows x 10 columns]
print(df_dop_null.info())
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2398 non-null   float64
#  1   고가      2398 non-null   float64
#  2   저가      2398 non-null   float64
#  3   거래량     2398 non-null   float64
#  4   금액(백만)  2398 non-null   float64
#  5   개인      2398 non-null   int64
#  6   기관      2398 non-null   int64
#  7   외인(수량)  2398 non-null   int64
#  8   외국계     2398 non-null   int64
#  9   외인비     2398 non-null   float64
#  10  종가      2398 non-null   float64
# dtypes: float64(7), int64(4)

#numpy 저장
SSD_data = df_dop_null.to_numpy()
print(SSD_data)
print(type(SSD_data)) # <class 'numpy.ndarray'>
print(SSD_data.shape) # (2398, 10)
np.save('./stock_pred/SSD_prepro_data2.npy', arr=SSD_data)