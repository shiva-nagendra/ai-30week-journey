#week 14 day 5
#Naive Bayes

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#load data
data = load_breast_cancer(as_frame=True)
df = data.frame

#train split and target
x = df.drop(columns=["target"])
y = df["target"]

#train test split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

#Naive bayes model
model_pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model",GaussianNB())
])


