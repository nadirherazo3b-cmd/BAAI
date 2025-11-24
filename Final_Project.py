import pandas as pd
from scipy.stats import zscore


df = pd.read_excel("Inditex_Simplified_ModelB.xlsx")
print("Null values per column:\n",df.isnull().sum())  #This shows how many values ​​are missing in each column. If all the sums are 0, everything is ready.


#Checking ranges for financial variables
# We define the columns to check as non-negative
financial_columns = [
    "Revenue (€ millions)", 
    "Net_Income (€ millions)", 
    "Earnings Per Share (EPS) €", 
    "Operating_Cash_Flow (€ millions)", 
    "Stock_Price (€)"
]
financial_out_of_range = (
    (df[financial_columns] < 0).sum()
)
print("Negative values per column:\n", financial_out_of_range) 

#Specific check for Market Return (can be positive or negative, but NOT zero)
market_return_null = df["Market_Return (%)"].isnull().sum()
print("Rows with null market_return: ", market_return_null)

# Detecting outliers using Z-score method
# Variables to analyze
outlier_vars = [
    "Stock_Price (€)", 
    "Revenue (€ millions)", 
    "Net_Income (€ millions)", 
    "Earnings Per Share (EPS) €", 
    "Operating_Cash_Flow (€ millions)"
]

# Calculate z-score for each variable
for var in outlier_vars:
    zs = zscore(df[var])
    print(f"\n{var}:")
    print("z-scores:", zs)
    # Rows that are potential outliers
    print("Possible Outliers years (|z| > 3):", df["Year"][abs(zs) > 3].tolist())


print(df.head())# Basic statistics for financial variables

stats = df.drop(columns=["Year"]).describe().loc[["mean", "std", "min", "max"]]
print(stats)

