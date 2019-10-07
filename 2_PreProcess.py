import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

rd = pd.read_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/0_rawdata-1.csv")
rd.columns = ["l","r","b","t","lat","lon","hea","ang","w","gt"]

# 데이터 중 bounding box 좌측변 좌표가 0이고 우측변좌표도 0인경우, 또는 우측변이 1에 해당하는 경우 주변차량 전체가 영상 내 인식되지 않은 것으로 판단하여
# 데이터셋에서 제외하였고, bounding box의 상하측변 y좌표, 주변차량의 local x,y좌표, 방위각, yaw angle은 현재 단계에서 활용하지 않아 데이터셋에서 제외
dataset = rd.drop(rd[((rd.l==0) & (rd.r==0))|(rd.r==1)].index)
dataset = dataset.drop(["b","t","lat","lon","hea","ang"],1)


dataset.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/1_dataset-1.csv",header=False,index=False)
