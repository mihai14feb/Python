import pandas as pd

# Simulam un fisier csv
data = {'Nume':['Ana','Bogdan','Cristi'], 'Varsta':[25,30,35], 'Salariu':[3000,4000,5000]}
df = pd.DataFrame(data)
print(df.head()) # afiseaza primele 5 randuri din Data Frame
print(df.describe()) # afiseaza statistici despre Data Frame (min, max, medie etc.)
print(df.info()) # afiseaza detalii despre tipurile de date din Data Frame