import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'iris_model.pkl')

test_df = pd.read_csv(BASE_DIR / 'Iris.csv')
test_df= test_df.drop('Id', axis=1)

X = test_df.drop("Species", axis=1)
Y = test_df["Species"]
pred = model.predict(X)
accuracy = accuracy_score(pred,Y)
print("Accuracy Score:", accuracy)

assert accuracy >= 0.81 #Threshold for accuracy score. If the accuracy score is less than 0.81, the test will fail and an AssertionError will be raised and the model will not deploy as a result.
print("Test Case 1 Passed: Accuracy is greater than or equal to 0.81")
assert len(pred) == len(X)
print("Test Case 2 Passed: Number of predictions is equal to number of input samples")