# Coded by Doctxing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pltcolor as pltcl
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_absolute_error
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.model_selection import train_test_split

class check:
    def __init__(self,infill,ifdiff=0):
        self.infill=infill
        self.ifdiff=ifdiff
    def process(self):
        df=pd.read_csv(self.infill)
        data=df.copy()
        data['time'] = pd.DataFrame({'time': range(1,len(data['DELM'])+ 1)})
        data['DELM']=data['DELM'].astype(float)
        data=data.set_index('time')
        data.info()
        plt.plot(data.index,data['DELM'].values)
        plt.show()

        split_index = int(0.9 * len(data))
        # 划分训练集
        train = data.iloc[:split_index]
        # 划分测试集
        test = data.iloc[split_index:]
        
        if self.ifdiff==1:
            #叉分
            diff_data = data[['time','DELM']].copy()
            diff_data.index=diff_data['time']
            # 一阶差分
            diff_data['diff_1'] =  diff_data.diff(1).dropna()
            # 二阶差分
            diff_data['diff_2'] = diff_data['diff_1'].diff(1).dropna()
            # 作图
            diff_data.plot(subplots=True,figsize=(18,20))

            plt.show()
            diff_data.to_csv('diffeddata.csv')
            # 对比选择几阶差分,肉眼观察数据平稳性
            diff_data = diff_data.fillna(0)
            # 单位根检验-ADF检验
            print(sm.tsa.stattools.adfuller(train['DELM']))
            #print(sm.tsa.stattools.adfuller(diff_data['diff_1']))
            #print(sm.tsa.stattools.adfuller(diff_data['diff_2']))

            #1%、%5、%10不同程度拒绝原假设的统计值和ADF比较，ADF同时小于1%、5%、10%即说明非常好地拒绝该假设，小于三个level的统计值，说明数据是平稳的
            #本数据还行
            # 白噪声检验
            print(acorr_ljungbox(train['DELM'], lags = [6, 12],boxpierce=True))
            #阶 延迟下LB和BP统计量的P值小于显著水平（α = 0.05 \alpha=0.05α=0.05）,此时拒绝序列为纯随机序列的原假设，认为该序列为非白噪声序列
            #可以通过AIC或BIC来确定最合适的阶数
            trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=20,max_ma=5)
            print('train AIC', trend_evaluate.aic_min_order)
            print('train BIC', trend_evaluate.bic_min_order)
            #order=(p,d,q)d为叉分阶数
        
        # 计算ACF
        acf=plot_acf(train['DELM'])
        plt.title("DELM自相关图")
        plt.show()
        # PACF
        pacf=plot_pacf(train['DELM'])
        plt.title("DELM偏自相关图")
        plt.show()
        # ACF	 PACF	 模型
        #拖尾	 截尾	  AR
        #截尾	 拖尾	  MA
        #拖尾	 拖尾	  ARMA
        
        model = sm.tsa.arima.ARIMA(endog=train['DELM'],order=(1,0,4))#eg
        arima_res=model.fit()
        arima_res.summary()
        #模型预测
        predict=arima_res.predict(train.index[0],train.index[-1])
        plt.plot(train.index,train['DELM'])
        plt.plot(train.index,predict)
        plt.legend(['y_true','y_pred'])
        plt.show()
        print(len(predict))
        #模型评价
        predict=arima_res.predict(test.index[0],test.index[-1])
        mean_absolute_error(test['DELM'],predict)
        #残差分析
        res=test['DELM']-predict
        residual=list(res)
        plt.plot(residual)
        #查看残差的均值是否在0附近
        np.mean(residual)
        #残差正态性检验

        plt.figure(figsize=(10,5))
        ax=plt.subplot(1,2,1)
        sns.distplot(residual,fit=stats.norm)
        ax=plt.subplot(1,2,2)
        res=stats.probplot(residual,plot=plt)
        plt.show()
        #将预测范围调大
        predict=arima_res.predict(train.index[-5],test.index[0]+10)
        plt.plot(train.index,train['DELM'])
        plt.plot(range(train.index[-5],train.index[-5]+len(predict)),predict)
        plt.legend(['y_true','y_pred'])
        plt.show()
        print(len(predict))

class generate:
    def __init__(self,infill,ifplt=1):
        self.infill=infill
        self.ifplt=ifplt
        
    def process(self):
        df=pd.read_csv(self.infill)
        data=df.copy()
        data['time'] = pd.DataFrame({'time': range(1,len(data['DELM'])+ 1)})
        data['DELM']=data['DELM'].astype(float)
        data=data.set_index('time')
        data.info()
        test = data.iloc[int(0.85 * len(data))-1:int(0.85 * len(data))+9]
        data = data.iloc[:int(0.85 * len(data))]
        
        
        model = sm.tsa.arima.ARIMA(endog=data['DELM'],order=(1,0,4))#eg
        arima_res=model.fit()
        arima_res.summary()
        
        if self.ifplt==1:
            # 创建图形对象
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
            custom_palette = sns.color_palette("RdBu", n_colors=10)
            sns.set(style="whitegrid", palette=custom_palette)

            # 绘制自相关图
            acf = plot_acf(data['DELM'], ax=ax1)
            ax1 = pltcl.plot_acf_colors(ax=ax1,markercolor=custom_palette[1], linecolor="gray", facecolor=custom_palette[2], barcolor=custom_palette[0])
            ax1.set_title("DELM-ACF")

            # 绘制偏自相关图
            pacf = plot_pacf(data['DELM'], ax=ax2)
            ax2 = pltcl.plot_acf_colors(ax=ax2,markercolor=custom_palette[8], linecolor="gray", facecolor=custom_palette[7], barcolor=custom_palette[9])
            ax2.set_title("DELM-PACF")

            ax1.set_facecolor('#f5f5f5')
            ax2.set_facecolor('#f5f5f5')
            
            # 调整子图之间的间距
            #plt.tight_layout()
            
            # 显示图形
            plt.savefig('sduvivnsvodscvnodvsid.png')
            plt.show()
        
        data = data.iloc[int(0.45 * len(data)):]
        #hua
        forecast = arima_res.get_forecast(steps=10)  # 假设您要预测未来 10 个时间点
        forecast_mean = forecast.predicted_mean
        forecast_se = forecast.se_mean

        # 计算置信区间
        confidence_interval = pd.DataFrame(index=forecast_mean.index)
        confidence_interval['lower'] = forecast_mean - 1.96 * forecast_se  # 90% 置信水平
        confidence_interval['upper'] = forecast_mean + 1.96* forecast_se  # 90% 置信水平

        if self.ifplt==1:
            
            custom_palette = sns.color_palette("RdBu", n_colors=10)
            sns.set(style="whitegrid", palette=custom_palette)

            # 创建图形对象和子图
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            # 绘制第一个图形
            ax1 = axes[0]
            ax1.plot(data.index, data['DELM'], label='Actual', color=custom_palette[9])  # 绘制实际数据
            ax1.plot(forecast_mean.index, forecast_mean.values, color=custom_palette[0], label='Forecast')  # 绘制预测值
            ax1.fill_between(confidence_interval.index, confidence_interval['lower'], confidence_interval['upper'], color=custom_palette[4], alpha=0.5, label='90% Confidence Interval')  # 绘制置信区间
            ax1.set_facecolor('#f5f5f5')
            ax1.legend(['True', 'Pred'])
            ax1.set_title('Visualization 1')

            # 绘制第二个图形
            ax2 = axes[1]
            ax2.plot(test.index, test['DELM'], label='Actual', color=custom_palette[9])  # 绘制实际数据
            ax2.plot(forecast_mean.index, forecast_mean.values, color=custom_palette[0], label='Forecast')  # 绘制预测值
            ax2.fill_between(confidence_interval.index, confidence_interval['lower'], confidence_interval['upper'], color=custom_palette[4], alpha=0.5, label='90% Confidence Interval')  # 绘制置信区间
            ax2.set_facecolor('#f5f5f5')
            ax2.legend(['True', 'Pred'])
            ax2.set_title('Visualization 2')
            # 显示图形
            plt.show()
            
            data = data.iloc[int(0.5 * len(data)):]
            plt.figure(figsize=(16, 6))
            predict=arima_res.predict(data.index[0],data.index[-1])
            plt.plot(data.index,data['DELM'],color=custom_palette[0],linewidth=2)
            plt.plot(range(data.index[0],data.index[0]+len(predict)),predict,color=custom_palette[9],linestyle='--',linewidth=2)
            plt.legend(['m_true','m_pred'])
            plt.gca().set_facecolor('#f5f5f5')
            plt.show()
        
        for i in range(5):
            print(data['DELM'].values[-5+i+1]-data['DELM'].values[-5+i])
        for i in range(5):
            print(forecast_mean.values[i+1]-forecast_mean.values[i])
            #认为dDELM正的话为赢
            #认为dDELM负的话为输
            #在DELM突变的时候局势扭转可能性大        

if __name__=='__main__':
    generate('processed.csv').process()
    
    