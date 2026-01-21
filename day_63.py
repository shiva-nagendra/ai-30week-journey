#skewness, Transformations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("day_62.csv")
print("\nData:\n",df) 


#original distribution
plt.figure()
plt.hist(df["income"],bins=6)
plt.title("income(original)")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()



df["income_log"] = np.log(df["income"])

# Compare correlation before and after
print("\nCorrelation with score income original:\n")
print(df[["income","score"]].corr())

print("\ncorrelation with score(log income):\n")
print(df[["income_log","score"]].corr())

#log transformation (fix scew)
plt.figure()
plt.hist(df["income_log"], bins=6)
plt.title("income(log transformed)")
plt.xlabel("income")
plt.ylabel("Frequency")
plt.show()

