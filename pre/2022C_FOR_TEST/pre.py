from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

Db = pd.read_csv('DATABASE.csv')
x = [[0],[1],[2],[3]]

def train(x,y,prex):
    x = np.array(x)
    y = np.array(y).reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(x,y)
    prey = regressor.coef_*prex + regressor.intercept_
    return prey[0][0]

def look_back(obj,i,num):
    if num == 4:
        return True
    if i<0:
        obj = [0] * 4
        return False
    if Db.loc[i,'Gtdy'] != Db.loc[i,'Gtdy']:
        return look_back(obj,i-1,num)
    else:
        obj[3-num] = Db.loc[i,'Gtdy']
        return look_back(obj,i-1,num+1)
    
for i in range(3,Db.shape[0]):
    y = [Db.loc[i-3,'Btdy'],
         Db.loc[i-2,'Btdy'],
         Db.loc[i-1,'Btdy'],
         Db.loc[i,'Btdy']]
    Db.loc[i,'Bpdt'] = train(x,y,4)

for i in range(4,Db.shape[0]):
    if Db.loc[i,'Gtdy'] != Db.loc[i,'Gtdy']:
        continue
    obj = [0]*4
    if look_back(obj,i,0) == False:
        continue
    Db.loc[i,'Gpdt'] = train(x,obj,4)

Db.to_csv('DATABASE_1.csv',index=False)

