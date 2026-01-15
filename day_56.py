#Data handling
#pandas

import pandas as pd

df = pd.read_csv("week8_day2.csv")
print("\nRaw_data: \n", df)

#check missing values
print("\nMissing values: \n", df.isnull())

#count missing
print("\nMissing count: \n", df.isnull().sum())

#Drop rows with missing data
clean = df.dropna()
print("\nAfter dropping missing rows: \n", clean)

#fill missng income with avg:
df["income"] = df["income"].fillna(df["income"]).mean()

#fill missing score with avg
df["score"]= df["score"].fillna(df["score"]).mean()

#fill missing age with median
df["age"] = df["age"].fillna(df["age"]).median()

#After filling missing values
print("\nAfter filling missing values: \n", df)