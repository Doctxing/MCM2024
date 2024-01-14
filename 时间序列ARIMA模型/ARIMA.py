import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

df=pd.read_csv('./testdata.csv',parse_dates=['时间'])
df.info()
data=df.copy()
data=data.set_index('时间')
plt.plot(data.index,data['datadata'].values)
plt.show()
train=data.loc[:'2018/1/13 23:45:00',:]#eg
test=data.loc['2018/1/14 0:00:00':,:]#eg
#叉分
diff_data = data.copy()
diff_data.index=data.index
# 一阶差分
diff_data['diff_1'] =  diff_data.diff(1).dropna()
# 二阶差分
diff_data['diff_2'] = diff_data['diff_1'].diff(1).dropna()
# 作图
diff_data.plot(subplots=True,figsize=(18,20))
plt.show()
# 对比选择几阶差分,肉眼观察数据平稳性

# 单位根检验-ADF检验
print(sm.tsa.stattools.adfuller(train['datadata']))
#1%、%5、%10不同程度拒绝原假设的统计值和ADF比较，ADF同时小于1%、5%、10%即说明非常好地拒绝该假设，小于三个level的统计值，说明数据是平稳的
# 白噪声检验
acorr_ljungbox(train['datadata'], lags = [6, 12],boxpierce=True)
#阶延迟下LB和BP统计量的P值小于显著水平（α = 0.05 \alpha=0.05α=0.05）,此时拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列
# 计算ACF
acf=plot_acf(train['datadata'])
plt.title("data的自相关图")
plt.show()
# PACF
pacf=plot_pacf(train['datadata'])
plt.title("data的偏自相关图")
plt.show()
# ACF	 PACF	 模型
#拖尾	 截尾	  AR
#截尾	 拖尾	  MA
#拖尾	 拖尾	  ARMA
#可以通过AIC或BIC来确定最合适的阶数
trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=20,
                                            max_ma=5)
print('train AIC', trend_evaluate.aic_min_order)
print('train BIC', trend_evaluate.bic_min_order)
#order=(p,d,q)d为叉分阶数
model = sm.tsa.arima.ARIMA(train,order=(7,0,4))#eg
arima_res=model.fit()
arima_res.summary()
#模型预测
predict=arima_res.predict("2018/1/14 0:00:00","2018/1/14 23:45:00")
plt.plot(test.index,test['datadata'])
plt.plot(test.index,predict)
plt.legend(['y_true','y_pred'])
plt.show()
print(len(predict))
#模型评价
from sklearn.metrics import r2_score,mean_absolute_error
mean_absolute_error(test['datadata'],predict)
#残差分析
res=test['datadata']-predict
residual=list(res)
plt.plot(residual)
#查看残差的均值是否在0附近
np.mean(residual)
#残差正态性检验
import seaborn as sns
from scipy import stats
plt.figure(figsize=(10,5))
ax=plt.subplot(1,2,1)
sns.distplot(residual,fit=stats.norm)
ax=plt.subplot(1,2,2)
res=stats.probplot(residual,plot=plt)
plt.show()
#将预测范围调大
predict=arima_res.predict("2018/1/14 0:00:00","2018/1/18 23:45:00")

plt.plot(range(len(predict)),predict)
plt.legend(['y_true','y_pred'])
plt.show()
print(len(predict))

