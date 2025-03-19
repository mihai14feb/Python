import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Date simple: y = 2x + 1
x = np.array([0,1,2,3,4], dtype=float)
y = np.array([1,3,5,7,9], dtype=float)

# Model foarte simplu
model = models.Sequential([
    layers.Dense(1, input_shape=[1])
])

# Compileaza
model.compile(optimizer='sgd', loss='mean_squared_error')

# Antreneaza
model.fit(x, y, epochs=50, verbose=1)

print(f"Predicție pentru x=5: {model.predict(np.array([5]))[0][0]}")