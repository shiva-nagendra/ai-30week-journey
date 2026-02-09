#week 12 project

#project goal : Binary Classification using Logistic Regression, with proper evaluation:
# 	•	Accuracy
# 	•	Confusion Matrix
# 	•	Precision, Recall, F1
# 	•	ROC–AUC
# 	•	Clear interpretation (what improved, what traded off)

# Dataset: Breast Cancer (sklearn, offline)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#feature split and target
x = df.drop(columns=["target"])
y = df["target"]

#split test and train

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#model train
model = LogisticRegression(max_iter=3249)
model.fit(x_train,y_train)

#model evaluation
probs = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

acc = print("\nAccuracy score:", round(accuracy_score(y_test,y_pred),2))
cm = print("\nConfusion matrix:\n", confusion_matrix(y_test,y_pred))
prec_score = print("\nPrecision score:",round(precision_score(y_test,y_pred),2))
recall = print("\nRecall score:",round(recall_score(y_test,y_pred),2))
f1Score = print("\nF1 score:", round(f1_score(y_test,y_pred),2))
RocAuc = print("\nRoc_Auc score:",round(roc_auc_score(y_test,probs),2))

