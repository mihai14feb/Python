import tensorflow as tf
from tensorflow.keras import layers, models

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(60000,28,28,1) / 255.0 # 60k imagini, rezolutie 28x28, 1 canal de culori (alb negru)
test_images = test_images.reshape(10000,28,28,1) / 255.0 # 10k imagini pt test

model = models.Sequential([
    layers.Input(shape=(28,28,1)), # modernizam, in loc de input_shape scriem layers.Input, imaginea are 28x28
    layers.Conv2D(32,(3,3)), # cream 32 filtre 3x3
    layers.MaxPooling2D(2,2), # reduce dimensiunea de 2 ori (14x14)
    layers.Conv2D(64,(3,3)), # 64 filtre 3x3
    layers.MaxPooling2D(2,2), # 7x7
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # 20% din imagini vor fi ignorate pentru antrenament mai eficient
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0) # testeaza modelul
print(f"Acuratete: {test_acc*100:.2f}%")