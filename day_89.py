#week 14 day 3
# Week 14 Day 3
# Non-linear SVM using RBF kernel

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create non-linear dataset
X, y = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear SVM
linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear"))
])

linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)
linear_acc = accuracy_score(y_test, linear_pred)

