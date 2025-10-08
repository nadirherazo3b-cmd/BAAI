#
# Nadir, 2025/10/08
# File: Nadir
# This is snippet task
#
import os
import glob 
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
#df_with_total.to_excel('output.xlsx', index=False)  #Esto es para guardar el DataFrame con la fila de totales en un nuevo archivo Excel
# print(df_resultado.head())       #Esto es para ver las primeras filas del DataFrame del resultado

# import os  (con este codigo se puede ver la ruta del archivo comunicandose con windows)


##### Separation Verfication Step 1-3

# 1. Path of my folder
folder_path = r"C:\Users\Nadir\Desktop\BAAI"

# 2. Obtain the excels documents (.xlsx y .xls)
excel_files = glob.glob(os.path.join(folder_path, "*.xls*"))

# 3. Order by alphabetical order
excel_files.sort()

# 4. Check the list to ensure everything is correct
print("Lista de archivos encontrados:")
for f in excel_files:
    print(os.path.basename(f))


    #### Separation Step 4-5

    # Read first documents (Financial.xlsx)
financial_file = os.path.join(folder_path, "Financial.xlsx")
df_financial = pd.read_excel(financial_file)
print("\n📊 Financial.xlsx:")
print(df_financial.head())

# Read second document (output.xlsx)
output_file = os.path.join(folder_path, "output.xlsx")  # o Resultado.xlsx
df_output = pd.read_excel(output_file)
print("\n📊 Output.xlsx:")
print(df_output.head())

# Check if output.xlsx is the last file in the list
if os.path.join(folder_path, "output.xlsx") == excel_files[-1]:
    print("\n✅ Output.xlsx is the last file in the list.")
else:
    print("\n➡️ There are more files after Output.xlsx.")

    #