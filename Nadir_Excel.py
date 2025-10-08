#
# Nadir, 2025/10/08
# File: Nadir
# This is snippet task
#
import pandas as pd

# 1. Input
df = pd.read_excel('Financial.xlsx')

# 2. Process
numeric_sums = df.select_dtypes(include='number').sum()
print (numeric_sums)

df2 = pd.concat([df, pd.DataFrame(numeric_sums)], ignore_index=True)
#df2.to_excel('Financial_with_sums.xlsx', index=False)

#df_resultado = pd.read_excel('Financial_with_sums.xlsx')  #Esto es para leer el archivo Excel que contiene los resultados



# 3. Output
print(df2)
# print(df_resultado.head())       #Esto es para ver las primeras filas del DataFrame del resultado

# import os  (con este codigo se puede ver la ruta del archivo comunicandose con windows)
