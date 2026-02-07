#week 12 day 4

#Precision and recall metrics

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#features split and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=42
)

#model training
model = LogisticRegression(max_iter=3000)
model.fit(x_train, y_train)

#evaluation
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
precision__score = precision_score(y_test, y_pred)
recall__score = recall_score(y_test, y_pred)

print("\nconfusion matrix:\n", cm)
print("\nPrecision score:\n", round(precision__score,2))
print("\nrecall score:\n",round(recall__score,2))