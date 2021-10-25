import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as db

# 도미와 빙어의 길이와 무게 데이터 csv 파일로 가져오기
bream_length = pd.read_csv('/Users/dahyechoi/workspace/fishapp/fish_csv/bream_length.csv')
bream_weight = pd.read_csv('/Users/dahyechoi/workspace/fishapp/fish_csv/bream_weight.csv')
smelt_length = pd.read_csv('/Users/dahyechoi/workspace/fishapp/fish_csv/smelt_length.csv')
smelt_weight = pd.read_csv('/Users/dahyechoi/workspace/fishapp/fish_csv/smelt_weight.csv')

# 도미의 길이와 무게를 넘파이 타입으로 변경
bream_length = np.array(bream_length)
bream_weight = np.array(bream_weight)

# 빙어의 길이와 무게를 넘파이 타입으로 변경
smelt_length = np.array(smelt_length)
smelt_weight = np.array(smelt_weight)

# 도미와 빙어의 길이, 무게 일차원배열을 이차원배열로 변경
bream_data = np.column_stack((bream_length, bream_weight))
smelt_data = np.column_stack((smelt_length, smelt_weight))

# 도미와 빙어 데이터 쉐이프 확인하기
print(bream_data)
print(smelt_data)
print(bream_data.shape)
print(smelt_data.shape)

# 도미와 방어 데이터 시각화(matplotlib)
plt.scatter(bream_data[:,0], bream_data[:,1]) 
plt.scatter(smelt_data[:,0], smelt_data[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 빙어와 도미 데이터를 합치기
fish_data = np.vstack((bream_data, smelt_data))
print(fish_data.shape)

# 도미와 빙어를 구분하기 위한 타깃 데이터 추가
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

# 타깃 데이터를 물고기 데이터에 추가
fish_target = fish_target.reshape((49,-1))
print(fish_target)
print(fish_target.shape)

fishes = np.hstack((fish_data,fish_target))
print(fishes)

# 물고기 데이터 셔플하기
index = np.arange(49) # 35(도미), 14(빙어)
np.random.shuffle(index)
print(index)

# 훈련 데이터
train_input = fish_data[index[:35]] 
train_target = fish_target[index[:35]] 

# 검증 데이터
test_input = fish_data[index[35:]] 
test_target = fish_target[index[35:]] 

# 데이터 시각화(matplotlib)
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 판다스 데이터프레임에 저장
train = np.column_stack((train_input[:,0], train_input[:,1], train_target))
train_dataFrame = pd.DataFrame(train, columns=["train_length", "train_weight", "tran_target"])
test = np.column_stack((test_input[:,0], test_input[:,1], test_target))
test_dataFrame = pd.DataFrame(test, columns=["test_length", "test_weight", "test_target"])
print(train_dataFrame)
print(test_dataFrame)

# 마리아디비에 데이터 넣기
engine = db.create_engine("mariadb+mariadbconnector://pandas:pandas1234@127.0.0.1:3306/pandasdb")

def insert():
    train = train_dataFrame
    train.to_sql("train", engine, index=False, if_exists="replace")
    test = test_dataFrame
    test.to_sql("test",engine, index=False, if_exists="replace")

insert()