# Coded by Doctxing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('pancake.csv')

# 创建画布和子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

custom_colors = sns.color_palette("RdBu", n_colors=10)
# 绘制小提琴图
sns.violinplot(x='dDELM', data=df, ax=axs[0],width=0.5,color=custom_colors[9])
axs[0].set_facecolor('#f5f5f5')
axs[0].set_title('Violin Plot')

# 将数据分成四个部分
range_1 = df[(df['dDELM'] >= -2) & (df['dDELM'] < -1)].shape[0]
range_2 = df[(df['dDELM'] >= -1) & (df['dDELM'] < 0)].shape[0]
range_3 = df[(df['dDELM'] >= 0) & (df['dDELM'] < 1)].shape[0]
range_4 = df[(df['dDELM'] >= 1) & (df['dDELM'] <= 2)].shape[0]
# 计算未包含在前四个范围内的数据数量
other_range = df[(df['dDELM'] >= 2) | (df['dDELM'] <= -2)].shape[0]

# 绘制饼图
sizes = [range_1, range_2, range_3, range_4, other_range]
labels = ['-2 to -1', '-1 to 0', '0 to 1', '1 to 2', 'Other']

colors = [custom_colors[0], custom_colors[2], custom_colors[4], custom_colors[8], custom_colors[9]]

axs[1].pie(sizes, labels=None, colors=colors, autopct='%1.1f%%', startangle=140)
axs[1].axis('equal')  # 使饼图保持圆形
axs[1].set_title('Distribution of Data in Different Ranges')

# 添加图例
axs[1].legend(loc='upper right', labels=labels)


# 显示图形
plt.show()
