#Week 14 project: Spam Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

#Load Data
df = pd.read_csv("spam.csv")

#convert labels to numeric
df["label"] = df["label"].map({"ham": 0, "spam" : 1})

x = df["message"]
y = df["label"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#models dict
models = {
    "Random Forest" : Pipeline([
        ("vectorizer", CountVectorizer()),
        ("model",RandomForestClassifier(random_state=42))
    ]),

    "Naive Bayes" : Pipeline([
        ("vectorizer", CountVectorizer()),
        ("model", MultinomialNB())
    ]),

    "SVM (linear)" : Pipeline([
        ("vectorizer", CountVectorizer()),
        ("model", LinearSVC())
    ])
}

#Model train and prediction
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    pre_score = precision_score(y_test, y_pred)
    f1__score = f1_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)




print(f"\n{name} Accuracy score", round(acc,3))