from sklearn.datasets import load_iris
import joblib
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('tester.csv')
data = data.sample(frac=1)
X = data[['R_mean', 'R_std', 'R_sum','G_mean', 'G_std', 'G_sum', 'B_mean', 'B_std', 'B_sum', 'total_mean', 'total_std', 'total_sum']]
# Number of rows in X
X_rows = len(X.index)
X_train = X[:int(X_rows*0.8)]
X_test = X[int(X_rows*0.8):]
y = data['class']
y_train = y[:int(X_rows*0.8)]
y_test = y[int(X_rows*0.8):]
clf = joblib.load('finalized_model.sav')
predictions = clf.predict(X)
from sklearn.metrics import accuracy_score
print(accuracy_score(y, predictions))
conf_mat = confusion_matrix(y, predictions)
print(conf_mat)
conf_mat = pd.DataFrame(conf_mat, columns=['whiteSpace', 'cellSpace'], index=['whiteSpace', 'cellSpace'])
conf_mat.to_csv('tester_confusion_matrix.csv')
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.savefig('tester_confusion_matrix.png')

# Classification Report
cfr = classification_report(y, predictions)
print(cfr)
