import pandas as pd

data = {
    'Nume':['Ana','Bogdan','Cristi','Mihai','Vali'],
    'Varsta':[25,30,35,40,45],
    'Salariu':[3000,4000,5000,6000,7000]}

df = pd.DataFrame(data)
print(df[df['Salariu'] > 3500]) # filtram persoanele cu salariu peste 3500 si le afisam
df['Impozit'] = df['Salariu'] / 5 # cream o noua coloana la tabel care stocheaza impozitul
print(df)