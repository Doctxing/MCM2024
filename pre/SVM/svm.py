#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#载入数据集
data=pd.read_csv("./iris.csv")
X = data.drop("Species", axis = 1).drop(data.columns[0], axis=1)
y = data["Species"]

# 划分数据集为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#创建模型并拟合
'''
linear核: 使用线性内核，主要用于线性可分的情形，速度快。
rbf核函数: 高斯径向核函数，拟合能力强，但是容易过拟合，耗时长，但最为常用。
poly核函数: 多项式核函数，常用于文本分类。
sigmoid核函数: 来源于神经网络，可以等效成两层的神经网络，泛化能力强。
'''
'''
C: 惩罚系数越高，说明越不能容忍出现误差
'''
clf=SVC(kernel='linear',C=1.0,random_state=42)
clf.fit(X_train,y_train)

# 预测测试结果
y_pred=clf.predict(X_test)

# 计算准确率
accuracy = clf.score(X_test,y_test)
print(f"Accuracy: {accuracy}")