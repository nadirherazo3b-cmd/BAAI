#
# Nadir, 2025/10/22
# File: James_Correlation_Analysis.py
# Apply correlation analysis to business problems


#Data manipulation and analysis
import pandas as pd
#import numpy as np

#Data visualization
#import matplotlib.pyplot as plt
#import seaborn as sns


#Statiscal analysis 
from scipy import stats


# 1. Input
df = pd.read_csv('Correlation_Analysis_Data.csv')

# print(df.isnull().sum())  
# print(df.isnull().sum().sum())

# print("Data loaded sucessfully")
# print(f'dataframe shape{df.shape}')

correlation, p_value = stats.pearsonr(df['Marketing_Spend'], df['Sales_Revenue'])

print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("The correlation is statiscally significant!")

# print(df.head())
# print(df.tail())
# print(df.mean())
# print(df.describe())

# 2. Process
# df[df.Marketing_Spend >7500]
# df[df.Customer_Satisfaction > 9.1].count()

# missing_values = df.isnull().sum()
# print("Missing Values per Column:")
# print(missing_values)
# print(f"\nTotal missing values: {missing_values.sum()}")


# 3. Output