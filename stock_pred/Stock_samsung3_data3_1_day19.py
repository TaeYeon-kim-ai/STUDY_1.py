
import numpy as np
import pandas as pd

df = pd.read_csv("../data/csv/삼성전자.csv", index_col=0, header=0, encoding='CP949')
df1 = pd.read_csv("../data/csv/삼성전자0115.csv", index_col=0, header=0, encoding='CP949')
# print(df.shape)
# print(df.info())
# print(df.columns)
print(df.shape)
print(df.info())
print(df.columns)

#df1 전처리
df = df.drop(df.index[0])
print(df.shape)
print(df.shape)

df1 = df1.drop(['전일비','Unnamed: 6'], axis=1)
print(df1)
df1 = df1.iloc[[0,1,2]]
df = pd.concat([df1, df])
# print(df1)
# print(df.head())

# print(df) #[2401 rows x 14 columns]
# 80 rows x 14 columns]
#              시가       고가       저가       종가   등락률         거래량     금액(백만)   신용비     개인           기관      외인(수량)         외국계        프로그램    외인비일자
# 2021-01-15   89,800   91,800   88,000   88,000 -1.90  33,117,980  2,947,682  0.00   7,510,662   -4,949,552           0    -261,904  -3,522,801  55.57
# 2021-01-14   88,700   90,000   88,700   89,700  0.00  26,393,970  2,356,662  0.10   3,239,160   -5,859,292   2,922,186   2,193,784  -1,091,335  55.57
# 2021-01-13   89,800   91,200   89,100   89,700 -0.99  36,068,848  3,244,067  0.10   4,600,807   -1,794,818  -1,898,330  -2,774,590  -2,190,774  55.52
# 2021-01-12   90,300   91,400   87,800   90,600 -0.44  48,682,416  4,362,546  0.08   8,233,252   -5,885,518  -2,125,136  -2,093,652  -4,498,684  55.56
# 2021-01-11   90,000   96,800   89,500   91,000  2.48  90,306,177  8,379,238  0.09  18,882,865  -13,590,286  -5,439,782  -4,979,740  -6,795,684  55

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
y = df_sorted.iloc[:,0]
del df_sorted['시가']
df_sorted['시가'] = y
# print(df_sorted)
# print(df_sorted.columns)

# 3. 결측값이 들어있는 행 전체 제거
df_dop_null = df_sorted.dropna(axis=0)
print (df_dop_null.isnull().sum())
print(df_sorted.columns) 
print(df_sorted.tail())
#          0       1       2       3          4       5           6           7      8     9             10        11         12       13
# Index(['고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '시가'],

# 고가
a = df_dop_null.iloc[:1735,0] / 50
b = df_dop_null.iloc[1735:,0] 
df_dop_null['고가'] = pd.concat([a,b])
print(df_dop_null['고가'])
print(df_dop_null['고가'].shape)

# 저가
a = df_dop_null.iloc[:1735,1] / 50
b = df_dop_null.iloc[1735:,1]
df_dop_null['저가'] = pd.concat([a,b])
print(df_dop_null['저가'])
print(df_dop_null['저가'].shape)

# 종가
a = df_dop_null.iloc[:1735,2] / 50
b = df_dop_null.iloc[1735:,2]
df_dop_null['종가'] = pd.concat([a,b])
print(df_dop_null['종가'])
print(df_dop_null['종가'].shape)
print(df_dop_null)

# 거래량
a = df_dop_null.iloc[:1735,4] *50
b = df_dop_null.iloc[1735:,4]
df_dop_null['거래량'] = pd.concat([a,b])
print(df_dop_null['거래량'])
print(df_dop_null['거래량'].shape)

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

# 시가
a = df_dop_null.iloc[:1735,13] / 50
b = df_dop_null.iloc[1735:,13]
df_dop_null['시가'] = pd.concat([a,b])
print(df_dop_null['시가'])
print(df_dop_null['시가'].shape)

# # 5. 상관계수 확인
# print(df_dop_null.corr())
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# import seaborn as sns
# font_path = "C:/STUDY/font/NanumBarunpenB.ttf"
# font_name = font_manager.FontProperties(fname=font_path).get_name()
# plt.rc('font', family=font_name)
# sns.set(font_scale=1)#폰트크기
# sns.heatmap(data=df_dop_null.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 6. 열제거(분석하고자 하는것 남기기)
#          0       1       2       3          4       5           6           7      8     9             10        11         12       13
# Index(['고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '시가'],
del df_dop_null['등락률']
del df_dop_null['신용비']
del df_dop_null['기관']
del df_dop_null['외인(수량)']
del df_dop_null['외국계']
del df_dop_null['프로그램']
del df_dop_null['외인비']

print(df_dop_null)

#7. 최종 데이터 확인
print(df_dop_null.shape) #[2397 rows x 10 columns]
print(df_dop_null.info())
print(df_dop_null) #[2397 rows x 10 columns]
# Data columns (total 7 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   고가      1088 non-null   float64
#  1   저가      1088 non-null   float64
#  2   종가      1088 non-null   float64
#  3   거래량     1088 non-null   float64
#  4   금액(백만)  1088 non-null   float64
#  5   개인      1088 non-null   int64
#  6   시가      1088 non-null   float64

#데이터 추출
df_dop_null = df_dop_null.iloc[1314:]
print(df_dop_null.tail()) #[2397 rows x 10 columns]
print(df_dop_null.shape) #[2397 rows x 10 columns]
print(df_dop_null.head()) #[2397 rows x 10 columns]

#numpy 저장
SSD_data = df_dop_null.to_numpy()
print(SSD_data)
print(type(SSD_data)) # <class 'numpy.ndarray'>
print(SSD_data.shape) # (1085, 7)
np.save('./stock_pred/SSD_prepro_data3.npy', arr=SSD_data)

# 2016-08-09  31580.0  31140.0  31340.0   9023200.0  283111.0  352300  31480.0
# 2016-08-10  31400.0  30680.0  30820.0  12340350.0  381460.0  926550  31340.0
# 2016-08-11  31180.0  30520.0  31180.0  10533600.0  325317.0  143050  30820.0
# 2016-08-12  31400.0  30880.0  30900.0  10474850.0  325885.0   62850  31180.0
# 2016-08-16  31520.0  30900.0  31360.0  10843100.0  339489.0  699050  30900.0

#제거
# 2018-05-03
# 2018-05-02
# 2018-04-30
