# Coded by Doctxing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

class judge:
    def __init__(self,data,ifdraw=1):
        self.data=data
        self.ifdraw=ifdraw
        
    def logistjudge(self):
        df=pd.read_csv(self.data).fillna(0)
        X=df[['DELM','dDELM']]     
        y=df['1ifwin'] # 假设这是一个二元变量，例如胜利或失败
        X = sm.add_constant(X)  # 添加常数项
        model = sm.Logit(y, X).fit()  # 使用 logistic 回归适用于二元变量
        # 打印模型摘要
        print(model.summary())
        ''' result:
                                Logit Regression Results
        ==============================================================================
        Dep. Variable:                 1ifwin   No. Observations:                  324
        Model:                          Logit   Df Residuals:                      321
        Method:                           MLE   Df Model:                            2
        Date:                Sun, 04 Feb 2024   Pseudo R-squ.:                 0.01760
        Time:                        08:46:04   Log-Likelihood:                -220.62
        converged:                       True   LL-Null:                       -224.57
        Covariance Type:            nonrobust   LLR p-value:                   0.01922
        ==============================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
        ------------------------------------------------------------------------------
        const         -0.0142      0.113     -0.126      0.900      -0.235       0.206
        DELM          -0.0679      0.025     -2.742      0.006      -0.116      -0.019
        dDELM         -0.0018      0.084     -0.022      0.983      -0.166       0.162
        ==============================================================================
        字段DELM显著性P>|z|值为0.006***，水平上呈现显著性，拒绝原假设，因此DELM会对1ifwin产生显著性影响，
        意味着DELM每增加一个单位，1ifwin为1.0的概率比0.0的概率低了6.571%。
        '''
        if self.ifdraw==1:
            coefficients = model.params
            p_values = model.pvalues
            custom_palette = sns.color_palette("RdBu", n_colors=10)
            sns.set(style="whitegrid", palette=custom_palette)

            # 创建两个子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            ax1.set_facecolor('#f5f5f5')
            ax2.set_facecolor('#f5f5f5')
            # 绘制系数
            coefficients.plot(kind='barh', ax=ax1, color=custom_palette[9], alpha=0.7)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title('Logistic Regression Coefficients')

            # 绘制 p 值的负对数
            ax2.barh(coefficients.index, -np.log10(p_values), color=custom_palette[0], alpha=0.7)
            ax2.set_xlabel('-log10(P-value)')
            ax2.set_title('Negative Log of p-values')

            # 显示图形
            plt.tight_layout()

            plt.show()
            
if __name__=='__main__':
    judge('processed.csv').logistjudge()