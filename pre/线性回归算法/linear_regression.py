##线性回归模型

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

##示例代码
if __name__ == '__main__':
    x = [0,1,2,3]
    y = [4,4,5,6]
    ##第一步，转换为np数组
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    ##第二步，调用训练集
    regressor = LinearRegression()
    regressor.fit(x,y)
    ##第三步，获取参数
    k = regressor.coef_ ##斜率
    b = regressor.intercept_ ##截距
    prex = 4
    prey = prex*k + b
    print(*prey) ##这个值是一维数组 输出结果为[6.5]