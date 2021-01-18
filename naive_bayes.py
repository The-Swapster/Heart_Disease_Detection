from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

df = pd.read_csv(r'C:\Users\HP\Documents\datasets\heart-disease-uci\heart.csv')
columns = df.columns
y = df['target']
X = df[columns]
X.drop('target', inplace = True, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred)
    return r[0]
print("sensitivity:", metrics.recall_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))
print("f1 score:", metrics.f1_score(y_test, y_pred))

df1 = pd.read_csv(r'C:\Users\HP\Documents\datasets\cardio.csv')
df1['age'] = df1['age']/365.25
df1['age'] = df1['age'].round()
columns1 = df1.columns
y1 = df1['cardio']
X1 = df1[columns1]
X1.drop('cardio', inplace = True, axis = 1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)

gnb1 = GaussianNB()
y_pred2 = gnb.fit(X_train1, y_train1).predict(X_test1)
