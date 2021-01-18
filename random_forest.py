import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

df = pd.read_csv(r'C:\Users\HP\Documents\datasets\heart-disease-uci\heart.csv')

columns = df.columns
y = df['target']
X = df[columns]
X.drop('target', inplace = True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators = 900)
model.fit(X_train, y_train)
predict = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predict))
print("Precision:", metrics.precision_score(y_test, predict))
print("Recall:", metrics.recall_score(y_test, predict))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred)
    return r[0]
print("sensitivity:", metrics.recall_score(y_test, predict))
print("specificity:", specificity_score(y_test, predict))
print("f1 score:", metrics.f1_score(y_test, predict))

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': list(range(1,1001))}
dt = RandomForestClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)
print("best params:", gs.best_params_)
print("best score:", gs.best_score_)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predict = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predict))
