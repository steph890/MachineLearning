import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


생존여부 = pd.read_csv("C:/Users/sanq.lee/Documents/STUDY/MachineLearning/example_titanic/gender_submission.csv")

트레인 = pd.read_csv("C:/Users/sanq.lee/Documents/STUDY/MachineLearning/example_titanic/train.csv")
테스트 = pd.read_csv("C:/Users/sanq.lee/Documents/STUDY/MachineLearning/example_titanic/test.csv")


# 데이터 살펴보기
생존여부
생존여부.shape
트레인
트레인.shape
테스트
테스트.shape

print(트레인.columns.values)
print(테스트.columns.values)

#트레인 데이터는 생존 여부가 결과로 포함되어 있으나, 테스트 데이터는 생존 여부가 없음 <- 이걸 우린 예측하는 것 => 로지스틱 회귀분석

#1. 결측치 확인 및 처리
트레인.isna().sum()     #Age 177, Cabin 687, Embarked 2
테스트.isna().sum()     #Age 86, Fare 1 , Cabin 327 개많음;;
# Age, Cabin => 결측치가 많음, 걍 버리면 안됨

트레인.dropna().shape
테스트.dropna().shape
#  결측치 날려버리면 데이터 다 날라감

트레인.describe(include=["O"])
# Ticket, Cabin은 전체 Count에 비해 Unique value가 너무 많으므로, 변수로서 영향을 주지 않을것 같음 => 제외

트레인2=트레인.drop(["Ticket", "Cabin","Embarked","Name"], axis=1)
# Name = PassengerId로 대체가능, Embarked는 결측치 2개 버림

트레인2['Age']=트레인2['Age'].fillna(트레인2['Age'].median());
# Age의 결측치를 median(중앙값) 으로 채웠습니다. 주어진 데이터가 결측치가 많기 때문에, 앞으로의 예측도 정확도가 떨어집니다.

트레인2['Sex'].replace("male",0,inplace=True)
트레인2['Sex'].replace("female",1,inplace=True)
# 범주형 자료를 숫자로 변환했습니다 male =>0, female =>1

pd.cut(트레인2['Age'], 5).unique()
트레인2['CategorizedAge']=pd.cut(트레인2['Age'], bins=[0,16,32,48,64,80],labels=['0','1','2','3','4'])
# 'Age' 데이터 연속형 -> 범주형 변환
# 임의로 5개 그룹을 지정

pd.cut(트레인2['Fare'], 5).unique()
트레인2['CategorizedFare']=pd.cut(트레인2['Fare'], bins=[-1,102,204,309,409,513],labels=['0','1','2','3','4'])
트레인2['Fare'].describe().sort_values();
트레인2.isna().sum()

트레인3=트레인2.drop(["Age", "Fare"], axis=1)
# 데이터 정리 완료

트레인4=트레인3.drop(["PassengerId","Survived"], axis=1)
타겟=트레인3["Survived"]
# 트레인 데이터는 변수만, 타겟 데이터는 결과만 남깁니다.

트레인4.isna().sum()


#테스트 데이터도 똑같은 과정을 거쳐서 정리합니다.
테스트2=테스트.drop(["Ticket", "Cabin","Embarked","Name"], axis=1)
테스트2['Age']=테스트2['Age'].fillna(테스트2['Age'].median());
테스트2['Sex'].replace("male",0,inplace=True)
테스트2['Sex'].replace("female",1,inplace=True)
테스트2['CategorizedAge']=pd.cut(테스트2['Age'], bins=[0,16,32,48,64,80],labels=['0','1','2','3','4'])
테스트2['CategorizedFare']=pd.cut(테스트2['Fare'], bins=[-1,102,204,309,409,513],labels=['0','1','2','3','4'])
테스트2['Fare'].describe().sort_values();
테스트2.isna().sum()
np.where(테스트2['CategorizedFare'].isna())
테스트2.iloc[152,] # <- 결측치 였음
테스트2=테스트2.dropna()


테스트3=테스트2.drop(["Age", "Fare"], axis=1)
# 데이터 정리 완료

테스트4=테스트3.drop("PassengerId", axis=1)

테스트4.isna().sum()


## 본격적인 학습 시작
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(트레인4, 타겟)

테스트3['Survived'] = lr.predict(테스트4)

## 문제 확인

테스트3.loc[테스트3['PassengerId']==892]
테스트3.loc[테스트3['PassengerId']==893]
테스트3.loc[테스트3['PassengerId']==894]
