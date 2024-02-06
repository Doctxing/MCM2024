import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# 读取文件
data = pd.read_csv("./winedata.csv")

#标准：
std=data["object"].unique().astype(str)
print(std)
 
# 将类别特征转换为数值
data = data.apply(lambda x: pd.Categorical(x).codes if x.dtype == "object" else x)
 

# 转换特征列名为字符串
data.columns = data.columns.astype(str)
 
# 分割数据为特征和目标
X = data.drop("object", axis = 1).drop(data.columns[0], axis=1)
y = data["object"]
 
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 创建决策树模型
model = DecisionTreeClassifier()
 
# 训练模型
model.fit(X_train, y_train)
 
# 预测
y_pred = model.predict(X_test)
 
# 计算模型准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
 
# 可视化决策树
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=data.columns[:-1],
    class_names=std,
    filled=True,
    rounded=True,
    special_characters=True,
)
 
graph = graphviz.Source(dot_data)
graph.render("./lenses_decision_tree")  # 将可视化图形保存为文件
graph.view()  # 在默认的图形查看器中打开可视化图形