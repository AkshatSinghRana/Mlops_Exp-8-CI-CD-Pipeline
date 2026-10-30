import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'D:\Semester_5\Mlops\exp_8\Iris.csv')
df = df.drop('Id', axis=1)
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('Accuracy:', accuracy)
joblib.dump(model, 'iris_model.pkl')
print('Model saved Successfully!!!')

