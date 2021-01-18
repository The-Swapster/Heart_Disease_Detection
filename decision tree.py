import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'C:\Users\HP\Documents\datasets\heart-disease-uci\heart.csv')
columns = df.columns
y = df['target']
X = df[columns]
X.drop('target', inplace = True, axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

param_grid = {
    'max_depth': list(range(1,1001))}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)
print("best params:", gs.best_params_)
print("best score:", gs.best_score_)

model = DecisionTreeClassifier()
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

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = columns.drop('target'),class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('heart.png')
Image(graph.create_png())

df1 = pd.read_csv(r'C:\Users\HP\Documents\datasets\cardio.csv')

columns1 = df1.columns

df1['age'] = df1['age']/365.25
df1['age'] = df1['age'].round()
y1 = df1['cardio']
X1 = df1[columns1]
X1.drop('cardio', inplace = True, axis = 1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X_train1, y_train1)
predict1 = model.predict(X_test1)

print("Accuracy:",metrics.accuracy_score(y_test1, predict1))
print("Precision:", metrics.precision_score(y_test1, predict1))
print("Recall:", metrics.recall_score(y_test1, predict1))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred)
    return r[0]
print("sensitivity:", metrics.recall_score(y_test1, predict1))
print("specificity:", specificity_score(y_test1, predict1))
print("f1 score:", metrics.f1_score(y_test1, predict1))

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = columns1.drop('cardio'),class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('heart1.png')
Image(graph.create_png())
