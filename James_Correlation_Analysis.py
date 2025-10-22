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
#from scipy import stats


# 1. Input
df = pd.read_csv("Simple_data.csv")

print(df.isnull().sum())

print("Data loaded sucessfully")
print(f'dataframe shape{df.shape}')




""""
print(df.head())
print(df.tail())
print(df.mean())
print(df.describe())
"""
# 2. Process
#df[df.Marketing_Spend >7500]
#df[df.Customer_Satisfaction > 9.1].count()
"""
missing values = df.isnull().sum()
print("Missing Values per Column:")
print(missing_values)
print(f"\nTotal missing values")
"""

# 3. Output