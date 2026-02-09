#week 12 day 6

#ROC Curve & AUC — judging a classifier without picking a threshold

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns = ["target"])
y = df["target"]

#split train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#model train
model = LogisticRegression(max_iter=3002)
model.fit(x_train, y_train)

#probs
prob = model.predict_proba(x_test)[:, 1]

#ROG_AUC score
auc = roc_auc_score(y_test, prob)
print("ROC - AUC:\n",round(auc,3))

#ROC curve
fpr, tpr, thresholds = roc_curve(y_test,prob)

plt.figure()
plt.plot(fpr, tpr, label = f"AUC = {round(auc,3)}")
plt.plot([0, 1], [0, 1], linestyle="--", color="red")
plt.title("ROC curve")
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend()
plt.show()

