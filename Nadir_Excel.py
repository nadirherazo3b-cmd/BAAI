#
# Nadir, 2025/10/08
# File: Nadir
# This is snippet task
#
import pandas as pd

# 1. Input
df = pd.read_excel('Financial.xls')
# 2. Process
sum = df.sum()

# 3. Output
print(df)