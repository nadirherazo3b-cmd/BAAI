#Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

#Input
df = pd.read_csv('Correlation_Analysis_Data.csv')

df.info()
# print(df.iloc[:,1:6])

#Process
#print(df.corr())
correlation_matrix = (df.iloc[:,1:6].corr())

print(correlation_matrix.round(3))

#Calculate correlation between Marketing_Spend and Sales_Revenue
# correlation, p_value = stats.pearsonr(
#     df['Marketing_Spend'],
#     df['Sales_Revenue']
# )

# print (f"Correlation coeffient: {correlation:.4f}")
# print (f"P-value: {p_value:.4e}")
# #en el e el numero negativo signiffica el numero de 0 qe hay despues de la coma y antes del numero

# if p_value <0.05:
#     print("The correlation is statistically significant!")

sns.heatmap(correlation_matrix)
plt.title('Nadir is the most intelligent person in the world')
plt.tight_layout()
plt.show()