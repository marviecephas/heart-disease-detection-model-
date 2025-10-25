importude = 'number')
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

print(f'accuracy = {accuracy} \n precision = {precision} \n recall = {recall} \n f1 = {f1_score}')
