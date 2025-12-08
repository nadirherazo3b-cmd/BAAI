#
# Nadir, 2025/11/19
# File: Inditex_Regression.py
# Perform regression analysis on Inditex data
#

#1. Input 
import pandas as pd
from scipy.stats import zscore
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.stats import shapiro, probplot
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson


# Load data
df = pd.read_excel("Inditex_Simplified_ModelB - Copy.xlsx")
# print("Null values per column:\n",df.isnull().sum()) 
df["log_Stock_Price"] = np.log(df["Stock_Price (€)"])
print(df.columns)

print("---------------------------------------------------------")

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
    #"Stock_Price (€)",
    "log_Stock_Price",
    "Operating_Cash_Flow (€ Billions) US format",
    "Market return",
    "Debt-to-equity",
    "Current Ratio",
    "Revenue Growth (%)",
]

cols = df[[
    #"Stock_Price (€)",
    "log_Stock_Price",
    "Operating_Cash_Flow (€ Billions) US format",
    "Market return",
    "Debt-to-equity",
    "Current Ratio",
    "Revenue Growth (%)",
]].describe().T
cols ['Missing Values'] = 0
cols ['Outliers'] = 0
# assign 2 outliers manually based on z-score analysis
cols.loc["log_Stock_Price", "Outliers"] = 2
cols.loc["Revenue Growth (%)", "Outliers"] = 2

final_table = cols[['count', 'mean', 'std', 'min', 'max', 'Missing Values', 'Outliers']]
print(final_table)

print("---------------------------------------------------------") #Separator

y = df["Stock_Price (€)"]
X = df[[

"Operating_Cash_Flow (€ Billions) US format",
"Market return",
"Debt-to-equity",
"Current Ratio",
"Revenue Growth (%)",

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
plt.figure(figsize=(10, 6))
df[var_cols].boxplot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Matrix

corr_matrix = X.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

print("---------------------------------------------------------")

# VIF (multicollinearity)

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

# Cook's Distance
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

# Plot Cook's Distance before regression
plt.figure(figsize=(8,4))
plt.bar(df.index, cooks_d, color='skyblue', edgecolor='black')
plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', label='4/n = {:.3f}'.format(4/len(cooks_d)))
plt.xlabel('Year')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance per year (Bar plot)")
plt.legend()
plt.tight_layout()
plt.show()

print("---------------------------------------------------------")

# Asymetría and Kurtosis
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

    # # Q–Q plot
    # plt.figure(figsize=(4,4))
    # probplot(data, dist="norm", plot=plt)
    # plt.title(f"Q-Q plot: {col}")
    # plt.tight_layout()
    # plt.show()

# Q-Q plot del log

# data = df["log_Stock_Price"].dropna()
# plt.figure(figsize=(4,4))
# probplot(data, dist="norm", plot=plt)
# plt.title("Q-Q plot: log_Stock_Price")
# plt.tight_layout()
# plt.show()

print("---------------------------------------------------------")

# Apply Logarithmic Transformation to Price (Recommended) Due to high skewness (skewness > 1)
#log for Stock_Price (right tail, always >0)

df["log_Stock_Price"] = np.log(df["Stock_Price (€)"])

check_cols = ["Stock_Price (€)", "log_Stock_Price"]

print(df[check_cols].skew().round(3))
print(df[check_cols].kurt().round(3))

print("---------------------------------------------------------")

# 2) Shapiro–Wilk for price and log price
for col in ["Stock_Price (€)", "log_Stock_Price"]:
    data = df[col].dropna()
    stat, p = shapiro(data)
    print(f"{col}: W = {stat:.3f}, p-value = {p:.3f}")

# Regression without log price
X_reg = sm.add_constant(X)
model = sm.OLS(y, X_reg).fit()
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

print("---------------------------------------------------------")

resid = model_log.resid           # Model residuals
mse = np.mean(resid**2)           # Mean Squared Error
rmse = np.sqrt(mse)               # Root Mean Squared Error

print("RMSE:", rmse)

print("---------------------------------------------------------")

# Assumptions Diagnostics (after regression)
# 1. Residuals vs fitted (linearity + homoscedasticity)
resid = model_log.resid
fitted = model_log.fittedvalues

plt.figure(figsize=(5,4))
plt.scatter(fitted, resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values (log_Stock_Price)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()

print("---------------------------------------------------------")

# 2. Breusch–Pagan Homoscedasticity Test
bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, model_log.model.exog)
print("Breusch-Pagan p-value:", bp_pvalue)

print("---------------------------------------------------------")

# 3. Normality of Residuals: Q-Q plot, Shapiro–Wilk test
#Shapiro-Wilk test
stat, p_shapiro = shapiro(resid) 
print("Shapiro-Wilk p-value:", p_shapiro)

#Q-Q plot
sm.qqplot(resid, line="45")
plt.title("Q-Q plot of regression residuals")
plt.tight_layout()
plt.show()

# Histograma + KDE
plt.figure(figsize=(5,4))
sns.histplot(resid, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of regression residuals")
plt.tight_layout()
plt.show()

print("---------------------------------------------------------")

#4. Durbin-Watson Test for Autocorrelation
dw = durbin_watson(resid)
print("Durbin-Watson:", dw)

print("---------------------------------------------------------")

#5. Variance Inflation Factor (VIF) after regression. Multicollinearity
X_const = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i)
                   for i in range(X_const.shape[1])]

print(vif_data)

print("---------------------------------------------------------")

# 6. Influential Observations:
# Cook's Distance
influence = OLSInfluence(model_log)
cooks_d = influence.cooks_distance[0]   

for i, d in enumerate(cooks_d):
    print(f"Obs {i}: Cook's D = {d:.3f}")

print("Max Cook's D:", np.max(cooks_d))

print("---------------------------------------------------------")

influence = OLSInfluence(model_log)
cooks_d = influence.cooks_distance[0]

n = len(cooks_d)
threshold = 4 / n


years = df['Year']          


plt.figure(figsize=(8,4))
plt.bar(years, cooks_d, color='skyblue', edgecolor='black')
plt.axhline(y=threshold, color='green', linestyle='--',
            label=f'4/n = {threshold:.3f}')
plt.xlabel('Year')
plt.ylabel("Cook\'s Distance")
plt.title("Cook\'s Distance per Year (model with log_Stock_Price)")
plt.legend()
plt.tight_layout()
plt.show()

# Box plot outliers for log_Stock_Price
# plt.figure(figsize=(4,6))
# sns.boxplot(y=df['log_Stock_Price'])
# plt.title('Boxplot de log_Stock_Price')
# plt.show()

print("---------------------------------------------------------")
