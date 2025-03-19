import tensorflow as tf
from tensorflow.keras import layers, models

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0 # pixelii sunt impartiti la 255 pentru a avea o valoare intre 0 si 9, ajuta antrenarea
test_images = test_images / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)), # vector intins
    layers.Dense(128, activation='relu'), # output 128, se utilizeaza relu
    layers.Dense(10, activation='softmax') # output 10, sotfmax e pentru probabilitati
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0) # testeaza modelul
print(f"Acuratete: {test_acc*100:.2f}%")