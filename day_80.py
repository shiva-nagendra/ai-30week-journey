#week 12 day 5
#F1 score

#F1 = 2 * (precision * recall) / (precision + recall)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns=["target"])
y = df["target"]

#split train and target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#model train
model = LogisticRegression(max_iter=4500)
model.fit(x_train, y_train)
 
#model evaluation
y_pred = model.predict(x_test)

#default prediction (threshold = 0.5)
print("\nPrecision score:\n", round(precision_score(y_test, y_pred),2))
print("\nRecall score:\n", round(recall_score(y_test, y_pred),2))
print("\nF1 score:\n", round(f1_score(y_test, y_pred),2))

# custom threshold
probs = model.predict_proba(x_test)[:, 1]
custom_prob = (probs>=0.7).astype(int)

print("\nWith threshold = 0.7")
print("Precision score:\n", round(precision_score(y_test, custom_prob),2))
print("Recall score:\n", round(recall_score(y_test, custom_prob),2))
print("F1 score:\n", round(f1_score(y_test, custom_prob),2))
