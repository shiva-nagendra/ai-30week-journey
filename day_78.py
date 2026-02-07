#week 12 day 3 confusion matrix

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#feature spilt nd target
x = df.drop(columns = ["target"])
y = df["target"]

#split train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#model selection
model = LogisticRegression(max_iter=2659)
model.fit(x_train, y_train)

#evaluation
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# confusion matrix 
#[[ TN   FP ]
#[ FN   TP ]]

print("\nAccuracy:", round(acc,2))
print("\nConfusion matrix:\n",cm)

