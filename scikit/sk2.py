import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4]])
y = np.array([2000,2500,3000,3500])

model = LinearRegression()
model.fit(x,y)

x_test = np.array([[5]])
pred = model.predict(x_test)
print(pred[0])

plt.scatter(x,y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.show()