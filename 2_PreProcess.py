import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

rd = pd.read_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/0_rawdata-1.csv")
rd.columns = ["l","r","b","t","lat","lon","hea","ang","w","gt"]

dataset = rd.drop(rd[((rd.l==0) & (rd.r==0))|(rd.r==1)].index)
dataset = dataset.drop(["b","t","lat","lon","hea","ang"],1)

# dataset['l1s'] = dataset['l'].shift(-10)
# dataset['l2s'] = dataset['l'].shift(-20)
# dataset['l3s'] = dataset['l'].shift(-30)
#
# dataset['r1s'] = dataset['r'].shift(-10)
# dataset['r2s'] = dataset['r'].shift(-20)
# dataset['r3s'] = dataset['r'].shift(-30)
#
# dataset['gt1s'] = dataset['gt'].shift(-10)
# dataset['gt2s'] = dataset['gt'].shift(-20)
# dataset['gt3s'] = dataset['gt'].shift(-30)
# dataset = dataset.dropna(axis=0)

dataset.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/1_dataset-1.csv",header=False,index=False)
#
# x = dataset[['l','r','w']].values
# y = dataset['3s'].values
#
# xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1,shuffle=False)
# trainx = pd.DataFrame(xtrain)
# testx = pd.DataFrame(xtest)
# trainy = pd.DataFrame(ytrain)
# testy = pd.DataFrame(ytest)
#
# trainx.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/2_xtrain.csv",header=False,index=False)
# testx.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/3_xtest.csv",header=False,index=False)
# trainy.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/2_ytrain.csv",index=False,header=False)
# testy.to_csv("C:/Users/JHYim/PycharmProjects/KSAE_2019_2/data/3_ytest.csv",index=False,header=False)

