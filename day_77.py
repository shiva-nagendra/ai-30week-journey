#week 12 day 2 Binary classificaton using logistic regression

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#feature split and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#model learning
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

acc_score = accuracy_score(y_test,y_pred)

print("Acuracy:\n",round(acc_score,2))

