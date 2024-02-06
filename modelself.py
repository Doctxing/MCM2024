# Coded by Doctxing

import pandas as pd
import numpy as np
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

#'./data/2023-wimbledon-1701.xlsx'
class modalef:
    def __init__(self,name,outname,ifdraw=1,ifsave=1):
        self.name=name
        self.outname=outname
        self.ifdraw=ifdraw
        self.ifsave=ifsave
    
    def genetato(self,deltamKk=1,deltaqk=1):
        
        def count(chars):
            is_same = (df_clear[chars] == df_clear[chars].shift()).astype(int)# 计算从头开始到当前位置相同元素的累积个数
            cumulative_count = is_same.cumsum()# 计算每个位置开始向前数到遇到与之相异的数据前的数据个数
            result = cumulative_count - cumulative_count.where(is_same != 1).ffill().fillna(0).astype(int)# 将结果写入 DataFrame 中的新列
            return result.apply(lambda x:x+1)
        
        def deltam(xi,victory,y,n):
            def K(x,victory):
                k1=1.35-0.35*np.exp(1-x)
                if victory==0:k2=-1 
                else: k2=1
                return np.around(deltamKk*k1*k2,decimals=4)
            def P(y,n):
                if y==0:p1,p2=0,0
                elif n==0: p1,p2=1.2,1
                else: p1,p2=1.2,-0.75
                return np.around((2-deltamKk)*p1*p2,decimals=4)
            return (K(xi,victory)+P(y,n))/2
    
        data=pd.read_excel(self.name)
        df=data.copy()  
        df.loc[df['p1_score']=='AD','p1_score'] = 50
        df.loc[df['p2_score']=='AD','p2_score'] = 50

        df_clear = df.drop(df[(df['p1_score'].astype(int)<=9) & (df['p1_score'].astype(int) >=1) & (df['p2_score'].astype(int)<=9) & (df['p2_score'].astype(int) >=1)].index)
        df_clear = df_clear.fillna(0)

        print(df_clear.isnull().sum())

        df_clear['1ifwin']=df_clear['point_victor'].apply(lambda x:2-x)#x
        df_clear['2ifwin']=df_clear['point_victor'].apply(lambda x:x-1)
        df_clear['1ifserver']=df_clear['server'].apply(lambda v:2-v)#victory
        df_clear['2ifserver']=df_clear['server'].apply(lambda v:v-1)

        df_clear['1count'] = count('1ifwin')
        df_clear['2count'] = count('2ifwin')
        df_clear['1runsum'] = df_clear['p1_distance_run'][::1].cumsum()
        df_clear['2runsum'] = df_clear['p2_distance_run'][::1].cumsum()
        df_1more=df_clear.apply(lambda x:x['1runsum']-x['2runsum'],axis=1)
        df_clear['1tirer'] =df_1more.apply(lambda x:x/df_1more.max())
        df_clear['2tirer'] =df_clear['1tirer'].apply(lambda x: -x)
        df_clear['if2']=df_clear['serve_no'].apply(lambda x:x-1)



        df_1deltaM = df_clear.apply(lambda x : deltam(xi=x['1count'],victory=x['1ifwin'],y=x['1ifserver'],n=x['if2']),axis=1)
        df_2deltaM = df_clear.apply(lambda x : deltam(xi=x['2count'],victory=x['2ifwin'],y=x['2ifserver'],n=x['if2']),axis=1)
        df_clear['1Mdata'] = df_1deltaM[::1].cumsum() + df_clear['1tirer'].apply(lambda x: deltaqk*x if np.fabs(x)<0.5 else deltaqk*0.5*np.sign(x))
        df_clear['2Mdata'] = df_2deltaM[::1].cumsum() + df_clear['2tirer'].apply(lambda x: deltaqk*x if np.fabs(x)<0.5 else deltaqk*0.5*np.sign(x))
        df_clear['DELM'] = df_clear['1Mdata']-df_clear['2Mdata']
        df_clear['DELM'] = df_clear['DELM'].apply(lambda x:x-df_clear['DELM'].mean())
        df_clear['preDELM'] = df_clear['DELM'].shift(1)
        df_clear['dDELM'] = df_clear['DELM'].diff(1).dropna()

        df_clear_timer1=df_clear[df_clear['game_victor']==1]
        df_clear_timer2=df_clear[df_clear['game_victor']==2]

        print(df_clear)

        if self.ifsave==1:
            df_clear[['elapsed_time','DELM','preDELM','dDELM','1ifwin']].to_csv(self.outname)

        if self.ifdraw==1:
            df_x=(df_clear['elapsed_time'])
            

            # 设置Seaborn的风格和调色板
            # 设置Seaborn的风格和调色板
            custom_palette = sns.color_palette("RdBu", n_colors=10)
            sns.set(style="whitegrid", palette=custom_palette)
            plt.figure(figsize=(16, 6))
            # 绘制图形  
            
            plt.plot(df_x, df_clear['DELM'], label='DELM',alpha=0.9,color='#585659',linewidth=1.7)
              # 调整平均线的透明度
            sns.scatterplot(data=df_clear_timer1, x='elapsed_time', y='DELM',s=120, marker='^',color=custom_palette[9], label='1win',alpha=0.6)
            sns.scatterplot(data=df_clear_timer2, x='elapsed_time', y='DELM',s=100, marker='o',color=custom_palette[0], label='2win',alpha=0.6)
            

            plt.xticks(df_x[::50])
            plt.legend()
            plt.gca().set_facecolor('#f5f5f5')
            plt.show()
        return df_clear['dDELM'],df_clear['1ifwin']

if __name__=='__main__':
    #modalef('./data/2023-wimbledon-1701.xlsx', 'processed.csv',ifdraw=1).genetato()
    os.chdir('./data')
    file_pattern = "2023-wimbledon-*.xlsx"
    file_names = glob.glob(file_pattern)
    for file_name in file_names:
        numeric_part = ''.join(char for char in file_name if char.isdigit())
        modalef(file_name, './processed/processed'+numeric_part[4:]+'.csv',ifdraw=0).genetato()
        