import numpy as np
import pandas as pd

df = pd.read_csv("../data/csv/KODEX 코스닥150 선물인버스.csv", index_col=0, header=0, encoding='CP949')

print(df.shape)
print(df.info())
print(df.columns)

#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   시가          1088 non-null   object
#  1   고가          1088 non-null   object
#  2   저가          1088 non-null   object
#  3   종가          1088 non-null   object
#  4   전일비         1088 non-null   object
#  5   Unnamed: 6  1088 non-null   object
#  6   등락률         1088 non-null   float64
#  7   거래량         1088 non-null   object
#  8   금액(백만)      1088 non-null   object
#  9   신용비         1088 non-null   float64
#  10  개인          1088 non-null   object
#  11  기관          1088 non-null   object
#  12  외인(수량)      1088 non-null   object
#  13  외국계         1088 non-null   object
#  14  프로그램        1088 non-null   int64
#  15  외인비         1088 non-null   float64

# 전처리
df = df.drop(['전일비', 'Unnamed: 6'], axis = 1)
print(df)
print(df.info())

#                 시가      고가      저가      종가   등락률         거래량   금액(백만)    신용비          개인         기관      외인(수량) 외국계  프로그램   외인비
# 일자
# 2021/01/15   4,420   4,545   4,405   4,515  2.27  58,448,052  262,267   0.00   1,309,200   -457,634           0   0     0  4.52
# 2021/01/14   4,400   4,440   4,380   4,415  1.03  36,237,676  159,885  10.15  -2,268,716  2,071,698     143,318   0     0  4.54
# 2021/01/13   4,370   4,410   4,335   4,370  0.00  37,052,864  162,112   9.77    -364,055   -127,739     510,104   0     0  4.40
# 2021/01/12   4,380   4,465   4,330   4,370  0.34  65,662,441  288,603   9.24     631,038   -836,132     148,229   0     0  3.82
# 2021/01/11   4,340   4,480   4,250   4,355  0.00  76,036,439  332,995   9.85   1,398,522    434,306  -1,598,665   0     0  3.59
# [1088 rows x 14 columns]

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
# df['프로그램'] = df['프로그램'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
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
y = df_sorted.iloc[:,0]
del df_sorted['시가']
df_sorted['시가'] = y
print(df_sorted)
print(df_sorted.columns)

# # 3. 결측값이 들어있는 행 전체 제거
# df_dop_null = df_sorted.dropna(axis=0)
# print (df_dop_null.isnull().sum())
# print(df_sorted.columns) 

# 3. 결측값이 들어있는 행 전체 제거
# 특정 행 제거
# 2018-05-03
# 2018-05-02
# 2018-04-30
df_dop_null = df_sorted.drop(['2018/05/03', '2018/05/02' , '2018/04/30'])
print(df_sorted.columns) 

#          0       1       2       3          4       5           6           7      8     9             10        11         12       13
# Index(['고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '시가'],

# 5. 상관계수 확인
print(df_dop_null.corr())
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
font_path = "C:/STUDY/font/NanumBarunpenB.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
sns.set(font_scale=1)#폰트크기
sns.heatmap(data=df_sorted.corr(), square=True, annot=True, cbar=True)
plt.show()

del df_dop_null['등락률']
del df_dop_null['신용비']
del df_dop_null['기관']
del df_dop_null['외인(수량)']
del df_dop_null['외국계']
del df_dop_null['프로그램']
del df_dop_null['외인비']

print(df_dop_null.info())
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   고가      1088 non-null   int64
#  1   저가      1088 non-null   int64
#  2   종가      1088 non-null   int64
#  3   거래량     1088 non-null   int64
#  4   금액(백만)  1088 non-null   int64
#  5   개인      1088 non-null   int64
#  6   시가      1088 non-null   int64
# dtypes: int64(7)

#numpy 저장
print(df_dop_null.tail()) #[2397 rows x 10 columns]
print(df_dop_null.head()) #[2397 rows x 10 columns]

SSD_data = df_dop_null.to_numpy()
print(SSD_data)
print(type(SSD_data)) # <class 'numpy.ndarray'>
print(SSD_data.shape) # (1088, 7)
np.save('./stock_pred/SSD_prepro_ETF_data3.npy', arr=SSD_data)
