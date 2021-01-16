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
#                 시가      고가      저가      종가   등락률         거래량   금액(백만)    신용비          개인         기관      외인(수량) 외국계  프로그램   외인비
# 일자
# 2021/01/15   4,420   4,545   4,405   4,515  2.27  58,448,052  262,267   0.00   1,309,200   -457,634           0   0     0  4.52
# 2021/01/14   4,400   4,440   4,380   4,415  1.03  36,237,676  159,885  10.15  -2,268,716  2,071,698     143,318   0     0  4.54
# 2021/01/13   4,370   4,410   4,335   4,370  0.00  37,052,864  162,112   9.77    -364,055   -127,739     510,104   0     0  4.40
# 2021/01/12   4,380   4,465   4,330   4,370  0.34  65,662,441  288,603   9.24     631,038   -836,132     148,229   0     0  3.82
# 2021/01/11   4,340   4,480   4,250   4,355  0.00  76,036,439  332,995   9.85   1,398,522    434,306  -1,598,665   0     0  3.59

