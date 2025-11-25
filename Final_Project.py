import pandas as pd
from scipy.stats import zscore
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

#Check transformations
variables = [
    "Stock_Price (€)",
    "Revenue (€ millions)",
    "Net_Income (€ millions)",
    "Earnings Per Share (EPS) €",
    "Operating_Cash_Flow (€ millions)"
]

# 1. Calculate bias for each variable
for var in variables:
    skew = df[var].skew()
    print(f"{var}: Skewness = {skew:.2f}")
    # If the bias is extreme, it is suggested to transform the variable
    if abs(skew) >= 1:
        print(f"Transforming {var} with natural logarithm due to high skewness.")
        # If there are zeros, add 1 before applying log
        df[f"log_{var}"] = np.log(df[var] + 1) #log_stock_price variable created

# Correlation Matrix and VIF
#Matrix of correlation
independent_vars = [
    "Revenue (€ millions)",
    "Net_Income (€ millions)",
    "Earnings Per Share (EPS) €",
    "Operating_Cash_Flow (€ millions)",
    "Market_Return (%)"
]
corr_matrix = df[independent_vars].corr()
print("Correlation matrix between independent variables:\n")
print(corr_matrix)


# VIF Calculation
# Drop rows with missing values for VIF calculation
X = df[independent_vars].dropna()
# Add constant term for intercept
X = sm.add_constant(X)
# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)



#print(df.head())# Basic statistics for financial variables

#stats = df.drop(columns=["Year"]).describe().loc[["mean", "std", "min", "max"]]
#print(stats)
