#1.2 멕시코풍 프랜차이즈 chipotle의 주문데이터 분석하기
#가장 많이 판매된 메뉴 Top 10은 무엇인가?
#메뉴별 판매량은 얼마나 될까?
#메뉴별 가격대는 얼마일까?

import pandas as pd

file_path = 'C:\data\python-data-analysis-master\data\chipotle.tsv'



#1. 데이터 (인사이트 발견을 위한 작업)
#read csv함수로 data불러오기

data = pd.read_csv(file_path, sep = '\t')

print(data.shape, data.info()) #(4622, 5)
# #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   order_id            4622 non-null   int64                        #주문번호
#  1   quantity            4622 non-null   int64                        #아이템의 주문 수량
#  2   item_name           4622 non-null   object                       #주문한 아이템의 이름
#  3   choice_description  3376 non-null   object #1246 개 결측값 존재   #주문한 아이템의 상세 선택 옵션
#  4   item_price          4622 non-null   object                       #주문 아이템의 가격정보
# dtypes: int64(2), object(3)

print(data.head(10))

#quantity 와 item_price의 수치적 특징 연속형 피처

#discribe()함수는 수치형 피처만 출력할 수 있음.
print(data.describe())

#order_id와 item_name의 개수
print(len(data['order_id'].unique())) #order_id의 row 갯수 출력
print(len(data['item_name'].unique())) #item_name의 row 종류 개수 출력

#1.1 탐색과 시각화하기
#가장 많이 주문한 아이템

#value_counts()적용
#가장 많이 주문한 아이템 top10출력
#item_name를 value_count()를 활용해 수량 출력 #10번째부터 그앞쪽으로 출력 0~9행 출력됨
#value_count() DataFrame['column]의 시리즈 함수에만 적용
item_count = data['item_name'].value_counts()[:10]
for idx, (val, cnt) in enumerate(item_count.iteritems(), 1):
    print("Top", idx, ":", val, cnt)

#아이템별 주문 개수와 총량
#groupby()함수 사용 : 데이터 프레임에서 특정 피처를 기준으로 그룹을 생성, 그룹별 연산 적용가능
order_count = data.groupby('item_name')['order_id'].count()
print(order_count[:10]) #아이템별 주문 개수를 출력함

#아이템 주문 총량 계산
item_quantity = data.groupby('item_name')['quantity'].sum()
item_quantity[:10]#아이템별 주문 총량 출력

#5.시각화
import numpy as np
import matplotlib.pyplot as plt

item_name_list = item_quantity.index.tolist()
x_pos = np.arange(len(item_name_list))
order_cut = item_quantity.values.tolist()

plt.bar(x_pos, order_cut, align = 'center')
plt.ylabel('order_item_count')
plt.title('Distribution of all orderd item')

plt.show()

#사용함수
#.unique() #범주형 함수 수량파악
#.groupby() #종목별 건당 수량파악
#.value_count() #row의 일정부분 출력

#=======================================================\

#1.2 데이터 전처리

print(data.info())
print(data['item_price'].head())

#column단위 데이터에 apply()함수로 전처리 적용
#.apply() : 시리즈 단위의 연산을 처리하는 기능 수행. .sum이나 .mean()과 같이 연산이 정의된 함수를 파라미터로 받음
#lambda 형식 :  (lambda x,y: x + y)(10, 20) 10+20 함수를 한줄로 만듬
#람다 함수를 사용해 x: 는 float로 전환 1:끝까지
data['item_price'] = data['item_price'].apply(lambda x: float(x[1:])) 
print(data.describe())

#주문당 평균 계산금액
data.groupby('order_id')['item_price'].sum().mean()

#한 주문에 10달러 이상 지불한 주문번호 id출력
data_orderid_group = data.groupby('order_id').sum()
results = data_orderid_group[data_orderid_group.item_price >= 10]
print(results[:10])
print(results.index.values)













