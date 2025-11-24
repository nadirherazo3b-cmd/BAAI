import pandas as pd

df = pd.read_excel("Inditex_Simplified_ModelB.xlsx")
print(df.isnull().sum())  #This shows how many values ​​are missing in each column. If all the sums are 0, everything is ready.

stats = df.drop(columns=["Year"]).describe().loc[["mean", "std", "min", "max"]]
print(stats)