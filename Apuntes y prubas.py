#print("Este es un cambio de prueba") 
#print ("Esta es otra prueba de que funciona git y esta sincronizado con vs code")
#print ("ultima comprobacion)")
#print ("prueba 27/09")
#print ("prueba 31/09)")

# Example operations in Python

# 1. Input
# X = 8
# Y = 3

# # 2. Process
# sumar = X + Y
# restar = X - Y
# multiplicar = X * Y
# division = X / Y
# division_entera = X // Y
# modulo = X % Y
# potencia = X ** Y

# # 3. Output
# print(f"Suma: {sumar}")
# print(f"Resta: {restar}")
# print(f"Multiplicación: {multiplicar}")
# print(f"División: {division}")
# print(f"División entera: {division_entera}")
# print(f"Módulo: {modulo}")
# print(f"Potencia: {potencia}")


# import os
# import glob 
# import pandas as pd

# # 1. Input Leer el archivo excel
# df = pd.read_excel('Inditex_Template.xlsx')

# # 2. Process
# sums = df.select_dtypes(include='number').sum()


# Optionally give a label for the row (e.g., 'Total')
#sums ['Name'] = 'Total'  #Add a value for the non-numeric colum

# Append the total row to the Data Frame
#df_with_total = pd.concat([df, pd.DataFrame([sums])], ignore_index=True)

#df_resultado = pd.read_excel('Financial_with_sums.xlsx')  #Esto es para leer el archivo Excel que contiene los resultados


# 3. Output
#print(df) ## Mostrar datos leídos

import pandas as pd
from scipy.stats import zscore
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.stats import shapiro, probplot


# Load your Excel file
# Update the filename to match your actual file name
# df = pd.read_excel("Inditex_Simplified_ModelB.xlsx")

# service_items = df[
#     ["Revenue (€ millions)", "Net_Income (€ millions)", "Earnings Per Share (EPS) €", "Operating_Cash_Flow (€ millions)"]
# ]

# item_var = service_items.var(axis=0, ddof=1)           # Item variances (columns)
# total_var = service_items.sum(axis=1).var(ddof=1)      # Variance of total score for each case

# alpha = len(service_items.columns) / (len(service_items.columns) - 1) * (
#     1 - item_var.sum() / total_var
# )
# print(round(alpha, 3))


# summary = df[cols].describe().loc[["count", "mean", "std", "min", "max"]]
# summary


df = pd.read_excel("Inditex_Simplified_ModelB - Copy.xlsx")
# print("Null values per column:\n",df.isnull().sum()) 
df["log_Stock_Price"] = np.log(df["Stock_Price (€)"])
print(df.columns)

# Check 1: Cash Flow positive
assert (df['Operating_Cash_Flow (€ Billions) US format'] > 0).all(), "CRITICAL ERROR: Negative Cash Flow was found."
print("Check 1: All cash flow is positive.")
# Check 2: Stock Price positive
assert (df['Stock_Price (€)'] > 0).all(), "CRITICAL ERROR: Negative or zero Stock Price found."
print("Check 2: All stock prices are positive.")
# Check 3 debt ratios non-negative
assert (df['Debt-to-equity'] >= 0).all(), "ERROR: Negative debt ratio found."
print("Check 3: Valid debt ratios.")
print("---------------------------------------------------------")

var_cols = [
    "Stock_Price (€)",
    "Operating_Cash_Flow (€ Billions) US format",
    "Market return",
    "Debt-to-equity",
    "Current Ratio",
    "Revenue Growth (%)",
]

cols = df[[
    "Stock_Price (€)",
    "Operating_Cash_Flow (€ Billions) US format",
    "Market return",
    "Debt-to-equity",
    "Current Ratio",
    "Revenue Growth (%)",
]].describe().T
cols ['Missing'] = 0
cols ['Outliers (|z|>3)'] = 0

final_table = cols[['count', 'mean', 'std', 'min', 'max', 'Missing', 'Outliers (|z|>3)']]
print(final_table)

print("---------------------------------------------------------")

y = df["Stock_Price (€)"]
X = df[[
# "Earnings Per Share (EPS) €",
"Operating_Cash_Flow (€ Billions) US format",
"Market return",
"Debt-to-equity",
"Current Ratio",
# "Revenue (€ Billions) US format",
# "ROE",
# "ROA"
"Revenue Growth (%)",
# "Inventory days"
]]

# Calculate z-score for each variable
for var in X:
    zs = zscore(df[var])
    print(f"\n{var}:")
    print("z-scores:", zs)
    # Rows that are potential outliers
    print("Possible Outliers years (|z| > 3):", df["Year"][abs(zs) > 3].tolist())

print("---------------------------------------------------------")

#Boxplot Outliers
# plt.figure(figsize=(10, 6))
# df[var_cols].boxplot()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# Matriz de correlación

corr_matrix = X.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

print("---------------------------------------------------------")

# VIF (multicolinealidad)

X_vif = sm.add_constant(X)

vif_data = []
for i in range(1, X_vif.shape[1]):
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append({"Variable": X_vif.columns[i], "VIF": vif})

vif_df = pd.DataFrame(vif_data)
print("VIF por variable:")
print(vif_df)

print("---------------------------------------------------------")

# Cook's Distance before regression
X_reg = sm.add_constant(X)
model_pre = sm.OLS(y, X_reg).fit()

# 3. Cook's Distance
influence = OLSInfluence(model_pre)
cooks_d, pvals = influence.cooks_distance

n = len(df)
threshold = 4 / n

cooks_df = pd.DataFrame({
    "Year": df["Year"],
    "Cooks_D": cooks_d,
    "Above_4_n": cooks_d > threshold
})

print("Cook's Distance per year:")
print(cooks_df)
print("\nUmbral 4/n =", threshold)
print("\nInfluential years (Cooks_D > 4/n):")
print(cooks_df[cooks_df["Above_4_n"]])

print("---------------------------------------------------------")

# Asymetría y Curtosis
skew_kurt = pd.DataFrame({
    "skewness": df[var_cols].skew(),      # asymetría (normal ~ 0)
    "kurtosis": df[var_cols].kurt()       # excess of kurtosis (normal ~ 0)
}).round(3)

print(skew_kurt)

print("---------------------------------------------------------")

# Normality Tests
vars_to_check = ["Stock_Price (€)", "Revenue Growth (%)"]

for col in vars_to_check:
    print(f"\nVariable: {col}")
    data = df[col].dropna()

    # Shapiro–Wilk
    stat, p = shapiro(data)
    print(f"Shapiro-Wilk W = {stat:.3f}, p-value = {p:.3f}")

    # Q–Q plot
    plt.figure(figsize=(4,4))
    probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q plot: {col}")
    plt.tight_layout()
    plt.show()

# data = df["log_Stock_Price"].dropna()

# # Q-Q plot del log

data = df["log_Stock_Price"].dropna()
plt.figure(figsize=(4,4))
probplot(data, dist="norm", plot=plt)
plt.title("Q-Q plot: log_Stock_Price")
plt.tight_layout()
plt.show()

print("---------------------------------------------------------")

# Apply Logarithmic Transformation to Price (Recommended) Due to high skewness (skewness > 1)
# log para Stock_Price (cola derecha, siempre >0)

df["log_Stock_Price"] = np.log(df["Stock_Price (€)"])

check_cols = ["Stock_Price (€)", "log_Stock_Price"]

print(df[check_cols].skew().round(3))
print(df[check_cols].kurt().round(3))

# # Regression
# X_reg = sm.add_constant(X)
# model = sm.OLS(y, X_reg).fit()
# print(model.summary())

print("---------------------------------------------------------")

#regression with log price
y_log = df["log_Stock_Price"]

# set of independent variables
X = df[[
    "Operating_Cash_Flow (€ Billions) US format",
    "Market return",
    "Debt-to-equity",
    "Current Ratio",
    "Revenue Growth (%)",
]]

X_reg_log = sm.add_constant(X)
model_log = sm.OLS(y_log, X_reg_log).fit()
print(model_log.summary())

# Cook's Distance after regression

# influence = OLSInfluence(model)
# cooks_d, pvals = influence.cooks_distance

# # Umbral típico: 4/n
# n = len(df)
# threshold = 4 / n

# cooks_df = pd.DataFrame({
#     "Year": df["Year"],
#     "Cooks_D": cooks_d,
#     "Above_4_n": cooks_d > threshold
# })

# print("Cook's Distance per year:")
# print(cooks_df)

# print("\nUmbral 4/n =", threshold)
# print("\nInfluential years (Cooks_D > 4/n):")
# print(cooks_df[cooks_df["Above_4_n"]])


# # Bar plot of Cook's Distance
# plt.figure(figsize=(8, 4))
# plt.bar(cooks_df["Year"], cooks_df["Cooks_D"], color="skyblue", edgecolor="black")
# plt.axhline(threshold, color="red", linestyle="--", label=f"4/n = {threshold:.3f}")
# plt.xlabel("Year")
# plt.ylabel("Cook's Distance")
# plt.title("Cook's Distance per year (Bar plot)")
# plt.legend()
# plt.tight_layout()
# plt.show()

print("---------------------------------------------------------")

#alpha de Cronbach
# def cronbach_alpha(df_items):
#     items = df_items.dropna()
#     k = items.shape[1]
#     var_items = items.var(axis=0, ddof=1)
#     total_score = items.sum(axis=1)
#     var_total = total_score.var(ddof=1)
#     alpha = (k / (k - 1)) * (1 - (var_items.sum() / var_total))
#     return alpha

# alpha_vars = df[[
#     "Earnings Per Share (EPS) €",
#     "Operating_Cash_Flow (€ millions)",
#     "Debt-to-equity",
#     "ROE",
#     "ROA"
# ]]

# alpha_value = cronbach_alpha(alpha_vars)
# print("Cronbach's alpha:", alpha_value)



