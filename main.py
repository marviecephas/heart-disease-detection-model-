from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
 sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/heart.csv'

df = pd.read_csv(file_path)
print(df.head(50))
print(df.info())
print(df.describe())

numerical_df = df.select_dtypes(include = 'number')
categorical_df = df.select_dtypes(include = 'object')

plt.figure(figsize = (10, 8))
sns.heatmap(numerical_df.corr(), annot= True, cmap = 'YlOrBr')

for col in numerical_df.columns :
  if col == 'HeartDisease':
    continue
  plt.figure(figsize = (10, 8))
  sns.histplot(data = df, x = col, hue = 'HeartDisease')

for col in categorical_df.columns :
  if col == 'HeartDisease':
    continue
  plt.figure(figsize = (10, 8))
  sns.histplot(data = df, x = col, hue = 'HeartDisease')

plt.show()

df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace = True)

plt.figure(figsize = (10, 8))
sns.histplot(data = df, x = 'Cholesterol', hue = 'HeartDisease')

plt.show()


df = pd.get_dummies(df, columns = categorical_df.columns, drop_first = True)


X = df.drop(columns = ['HeartDisease'])
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
logr = LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0,1])
cmd.plot(cmap = 'Blues')
plt.show()


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'accuracy = {accuracy} \n precision = {precision} \n recall = {recall} \n f1 = {f1}')
