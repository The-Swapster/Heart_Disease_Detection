import pandas as pd
import seaborn as sns

df = pd.read_csv(r'output.csv')
sns.barplot(x="Models", y="Accuracy",data=df[:9])
sns.barplot(x="Models", y="Precision",data=df[:9])
sns.barplot(x="Models", y="Recall",data=df[:9])
sns.barplot(x="Models", y="F1 Score",data=df[:9])

df1 = pd.read_csv(r'C:\Users\HP\Documents\research_paper\output_cardio.csv')
sns.barplot(x="Models", y="Accuracy",data=df1[:9])
sns.barplot(x="Models", y="Precision",data=df1[:9])
sns.barplot(x="Models", y="Recall",data=df1[:9])
sns.barplot(x="Models", y="F1 Score",data=df1[:9])
