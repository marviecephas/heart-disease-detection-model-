from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Import pandas

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
