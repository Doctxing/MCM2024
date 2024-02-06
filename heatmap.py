# Coded by Doctxing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
data = pd.read_csv('processed.csv')

data['1win']=data['1ifwin']
# 创建DataFrame
df = data[['DELM','dDELM','1win']]

# 计算相关系数
correlation_matrix = df.corr()

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Oranges', fmt=".2f",vmin=1, vmax=0.5)
plt.title('Correlation Heatmap')
plt.show()