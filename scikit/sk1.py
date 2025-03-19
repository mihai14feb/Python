import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Date de antrenament
x = np.array([[1],[2],[3],[4]]) # suprafata casei (m2)
y = np.array([100,200,300,400]) # pret (mii de lei)

# Model
model = LinearRegression()
model.fit(x,y)

# Predictie
x_test = np.array([[5]])
pred = model.predict(x_test)
print(pred[0])

# Vizualizare
plt.scatter(x,y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.show()