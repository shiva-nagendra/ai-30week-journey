
# Week 14 Day 1
# SVM Intuition Demo (Concept + Visualization)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Generate simple 2D dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Pipelines (scaling included because SVM needs it)
logistic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear"))
])

# Train models
logistic_pipe.fit(X, y)
svm_pipe.fit(X, y)

# Plotting decision boundaries
def plot_model(model, title):
    plt.figure()
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y)
    
    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z)
    plt.title(title)
    plt.show()

# Visualize
plot_model(logistic_pipe, "Logistic Regression Boundary")
plot_model(svm_pipe, "SVM Boundary (Max Margin)")

# Reflection:
# Logistic Regression separates classes with a straight boundary.
# SVM finds a boundary with maximum margin (largest separation).
# Both are linear here, but SVM focuses on geometric safety.