import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float)

model = models.Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=250, verbose=10)

prediction = model.predict(x)
for i in range(4):
    print(x[i], prediction[i][0], y[i])