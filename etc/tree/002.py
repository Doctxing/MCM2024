import pybaobabdt
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
 
data = arff.loadarff('vehicle.arff')
 
df = pd.DataFrame(data[0])
y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]
 
clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=0)
clf.fit(X, y)