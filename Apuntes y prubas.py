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


import os
import glob 
import pandas as pd

# 1. Input
df = pd.read_excel('Inditex_Template.xlsx')

# 2. Process
sums = df.select_dtypes(include='number').sum()


# Optionally give a label for the row (e.g., 'Total')
sums ['Name'] = 'Total'  #Add a value for the non-numeric colum

# Append the total row to the Data Frame
df_with_total = pd.concat([df, pd.DataFrame([sums])], ignore_index=True)

#df_resultado = pd.read_excel('Financial_with_sums.xlsx')  #Esto es para leer el archivo Excel que contiene los resultados


# 3. Output
print(df_with_total)
