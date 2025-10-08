#
# Nadir, 2025/10/08
# File: Nadir
# This is snippet task
#
import pandas as pd

# 1. Input
df = pd.read_excel('Financial.xlsx')

# 2. Process
sums = df.select_dtypes(include='number').sum()


# Optionally give a label for the row (e.g., 'Total')
sums ['Name'] = 'Total'  #Add a value for the non-numeric colum

# Append the total row to the Data Frame
df_with_total = pd.concat([df, pd.DataFrame([sums])], ignore_index=True)

#df_resultado = pd.read_excel('Financial_with_sums.xlsx')  #Esto es para leer el archivo Excel que contiene los resultados


# 3. Output
print(df_with_total)
df_with_total.to_excel('output.xlsx', index=False)  #Esto es para guardar el DataFrame con la fila de totales en un nuevo archivo Excel
# print(df_resultado.head())       #Esto es para ver las primeras filas del DataFrame del resultado

# import os  (con este codigo se puede ver la ruta del archivo comunicandose con windows)
