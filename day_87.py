# # Week 14 Day 1 — Support Vector Machine (SVM) Theory

# ## What is SVM?

# Support Vector Machine (SVM) is a supervised learning model used for classification.

# Its goal:

# Draw a boundary between classes with the **maximum possible margin**.

# ---

# ## What is Margin?

# Margin = Distance between the decision boundary and the nearest data points.

# SVM does not just separate classes.

# It finds the boundary that is:
# - Farthest from both classes
# - Most stable
# - Most confident

# This makes SVM robust.

# ---

# ## How SVM Thinks

# Logistic Regression:
# - Optimizes probability.
# - Draws a separating line.

# SVM:
# - Optimizes geometric distance.
# - Draws a boundary with the largest cushion.

# ---

# ## Why Scaling Is Mandatory

# SVM depends on distance calculations.

# If one feature has larger scale:
# - It dominates distance.
# - Boundary becomes distorted.

# Therefore:
# Always use StandardScaler with SVM.

# ---

# ## What Are Support Vectors?

# Support vectors are the data points closest to the boundary.

# They determine:
# - Where the boundary is placed
# - How wide the margin is

# If you remove non-support points,
# the boundary stays almost the same.

# ---

# ## When to Use SVM

# Use SVM when:
# - Dataset is medium-sized
# - Data is high-dimensional
# - Clear margin exists
# - You want strong classification performance

# ---

# ## Important Concepts (We’ll Expand Later)

# - Linear SVM → straight boundary
# - Kernel SVM → curved boundary
# - Hard margin vs Soft margin
# - C parameter → controls strictness

# ---

# ## Summary

# SVM is a classification model that:

# - Maximizes margin
# - Relies heavily on scaling
# - Works well in high dimensions
# - Uses support vectors to define the boundary