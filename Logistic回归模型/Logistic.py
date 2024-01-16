import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#使用数据集X，y
X=np.array(pd.read_csv('./x.csv',usecols=['0','1']))
y=np.array(pd.read_csv('./y.csv',usecols=['0']))
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建Logistic回归模型
logreg = LogisticRegression()

# 训练模型
logreg.fit(X_train, y_train)

# 预测测试集
y_pred = logreg.predict(X_test)

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# eg
print("[-2,-1]的预测结果为：",logreg.predict([[-2,-1]]),sep='')

