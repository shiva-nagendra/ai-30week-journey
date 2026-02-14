#week 14 day 3
# Week 14 Day 3
# Non-linear SVM using RBF kernel

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

#create non-linear dataset
x, y = make_circles(n_samples=500, noise= 0.1, factor=0.4, random_state=42)

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#linear SVM
linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear"))
])

linear_svm.fit(x_train,y_train)
linear_pred = linear_svm.predict(x_test)
acc_linear = accuracy_score(y_test,linear_pred)

#non-linear SVM
rbf_svm = Pipeline([
    ("scaler",StandardScaler()),
    ("model", SVC(kernel="rbf"))
])

rbf_svm.fit(x_train,y_train)
rbf_pred = rbf_svm.predict(x_test)
acc_rbf = accuracy_score(y_test,rbf_pred)


print("Linear SVM Accuracy:", round(acc_linear, 3))
print("RBF Kernel SVM Accuracy:", round(acc_rbf, 3))


# Reflection:
# Linear SVM struggles on non-linear data.
# RBF kernel allows curved boundaries.
# Kernel transforms space to make separation possible.