from sklearn.datasets import load_iris
import joblib
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


# Tree
plt.figure(figsize=(10,10))
plot_tree(clf, filled=True)
plt.savefig('tree.png')

# Predictions
predictions = clf.predict(X)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y, predictions))
filename = 'finalized_model.sav'
joblib.dump(clf, filename)
conf_mat = confusion_matrix(y, predictions)
print(conf_mat)
conf_mat = pd.DataFrame(conf_mat, columns=['whiteSpace', 'cellSpace'], index=['whiteSpace', 'cellSpace'])
conf_mat.to_csv('confusion_matrix.csv')
plt.figure(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.savefig('confusion_matrix.png')

# Classification Report
cfr = classification_report(y, predictions)
print(cfr)
