# Coded by Doctxing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.metrics import r2_score
from BETA import Alpha as ap
from sklearn.metrics import accuracy_score


file_pattern = "processed*.csv"
file_names = glob.glob(file_pattern)

accuracies = []

# 迭代每个文件进行独立的训练和测试
for file_name in file_names:
    df = pd.read_csv(file_name)
    
    train_data = []
    # 跳过当前文件名，将其他文件的数据加载到训练集中
    other_files = [f for f in file_names if f != file_name]
    for other_file in other_files:
        other_df = pd.read_csv(other_file)
        train_data.append(other_df)
    
    # 合并所有其他文件的数据
    df_train = pd.concat(train_data)
    
    
    df['dDELM']=df['DELM'].diff().fillna(0)
    df_train['dDELM']=df_train['DELM'].diff().fillna(0)
    df = df.dropna()
    df_train = df_train.dropna()
    # 提取特征和标签
    X = df
    X_train = df_train
    y = df['DELM']
    y_train = df_train['DELM']
    clf = ap(ifplt=0)
        
    # 训练模型
    clf.fit(X_train, y_train)
        
    # 预测并评估模型
    y_pred = clf.predict(X)
    # 利用方法计算accuracy
    accuracy = r2_score(y, y_pred)
    #accuracy = accuracy_score(y, y_pred)
    accuracies.append(accuracy)


for i, file_name in enumerate(file_names):
    print(f"File: {file_name}, Accuracy: {np.around(accuracies[i],decimals=4)}")

# 绘制柱状图
import seaborn as sns


# 假设 accuracies 和 file_names 是你的数据
# accuracies = [0.8, 0.85, 0.75, 0.9]
# file_names = ['File1', 'File2', 'File3', 'File4']
shortened_file_names = [file_name[-8:] for file_name in file_names]
plt.figure(figsize=(16, 8))

# 使用 Seaborn 的 color_palette 函数生成 BURD 渐变色调色板
colors = sns.color_palette("RdBu", len(file_names))
custom_palette = sns.color_palette("RdBu", n_colors=10)
sns.set(style="whitegrid")


# 使用 barplot 函数绘制柱状图，并设置颜色为 BURD 渐变色
sns.barplot(x=shortened_file_names, y=accuracies, palette=colors)

# 绘制折线图
plt.plot(range(len(accuracies)), accuracies, marker='^', color=custom_palette[9])

# 添加水平参考线
plt.axhline(y=np.mean(accuracies), color=custom_palette[9], linestyle='--',label='avg')

# 添加标题和标签
plt.xlabel('File')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores for Each Dataset')

# 设置 x 轴刻度标签旋转
plt.xticks(rotation=20)

# 设置 y 轴的范围
plt.ylim(0.7, 1.1)

# 显示图例
plt.legend()
plt.gca().set_facecolor('#f5f5f5')
# 显示图表
plt.show()


