#week 14 day 6
#compare multiple classification models

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#split features and target
x = df.drop(columns=["target"])
y = df["target"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

#models dict
models = {
    "Logistic regression" : Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3455))
    ]),

    "svm (linear)" : Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear"))
    ]),

    "KNN" : Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision tree" : DecisionTreeClassifier(random_state=42),

    "Naive Bayes":  GaussianNB()
    
}

#train and evaluate all
for name, model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print(f"\n{name} accuracy score:", round(acc,3))

print("\nReflection:")
print("Distance-based models require scaling.")
print("Tree-based models do not require scaling.")
print("Naive Bayes uses probability assumptions.\n")