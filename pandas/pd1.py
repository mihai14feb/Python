import pandas as pd

# Serie (vector cu index, asemanator unui dictionar)
seria = pd.Series([1,2,3,4], index=['a','b','c','d'])
print(seria)

# Data Frame (matrice 2dimensionala, asemanatoare unui tabel)
data = {'Nume': ['Ana', 'Bogdan'], 'Varsta': [25, 30]}
df = pd.DataFrame(data)
print(df)