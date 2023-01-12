from sklearn.datasets import load_iris

from sklearn import tree
import pandas as pd


data = pd.read_csv('master.csv')
data = data.sample(frac=1)
X = data[['R_mean', 'R_std', 'R_sum','G_mean', 'G_std', 'G_sum', 'B_mean', 'B_std', 'B_sum', 'total_mean', 'total_std', 'total_sum']]
# Number of rows in X
X_rows = len(X.index)
X_train = X[:int(X_rows*0.8)]
X_test = X[int(X_rows*0.8):]
y = data['class']
y_train = y[:int(X_rows*0.8)]
y_test = y[int(X_rows*0.8):]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predictions
predictions = clf.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))