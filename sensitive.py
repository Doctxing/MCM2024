# Coded by Doctxing

import modelself as mos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 创建模型实例
ctl = mos.modalef('./data/2023-wimbledon-1701.xlsx', 'processed1701.csv', ifsave=0, ifdraw=0)

# 生成数据
Klines = np.linspace(0, 2, 20)
Qlines = np.linspace(0, 2, 20)
acc_kendall = []
acc_pearson = []

# 生成二维数组存储结果
results_kendall = np.zeros((len(Klines), len(Qlines)))
results_pearson = np.zeros((len(Klines), len(Qlines)))

# 生成 x_data
x_data = pd.DataFrame(columns=['dDELM', 'win'])

# 计算结果
for i, Kkk in enumerate(Klines):
    for j, Qqq in enumerate(Qlines):
        # 生成数据
        x_data['dDELM'], x_data['win'] = ctl.genetato(Kkk, Qqq)
        # 计算相关系数
        kendall_corr = x_data.corr(method='kendall').iloc[0, 1]
        pearson_corr = x_data.corr(method='pearson').iloc[0, 1]
        acc_kendall.append(kendall_corr)
        acc_pearson.append(pearson_corr)
        # 存储结果
        results_kendall[i, j] = kendall_corr
        results_pearson[i, j] = pearson_corr

# 创建包含两个子图的图形
fig, axs = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})

# 绘制第一个子图（左边）
ax1 = axs[0]
K, Q = np.meshgrid(Klines, Qlines)
surf1 = ax1.plot_surface(K, Q, results_kendall, cmap='RdBu')
ax1.set_xlabel('Kk')
ax1.set_ylabel('Qq')
ax1.set_zlabel('Kendall Correlation Coefficient')
ax1.set_title('Kendall Correlation Coefficient vs. Kk and Qq')

# 绘制第二个子图（右边）
ax2 = axs[1]
surf2 = ax2.plot_surface(K, Q, results_pearson, cmap='RdBu')
ax2.set_xlabel('Kk')
ax2.set_ylabel('Qq')
ax2.set_zlabel('Pearson Correlation Coefficient')
ax2.set_title('Pearson Correlation Coefficient vs. Kk and Qq')

# 显示图形
plt.show()
