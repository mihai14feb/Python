import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_images = train_images / 255.0 # 50k imagini, rezolutie 32x32, 3 canale de culori (rgb)
test_images = test_images / 255.0 # 10k imagini pt test

datagen = ImageDataGenerator( # produce modificari aleatorii pt imagini
    rotation_range=30, # roteste imaginile aleatoriu cu pana la 30 de grade
    fill_mode='nearest',
    width_shift_range=0.3, # deplaseaza imaginea pe orizontala cu pana la 30% din latime
    height_shift_range=0.3, # la fel dar pe lungime
    brightness_range=[0.8,1.2], # luminozitate
    zoom_range=0,3, # zoom 30%
    horizontal_flip=True # inverseaza imaginea pe orizontala
)
datagen.fit(train_images)

model = models.Sequential([
    layers.Input(shape=(32,32,3)), # modernizam, in loc de input_shape scriem layers.Input
    layers.Conv2D(32,(3,3), activation='relu', padding='same'), # cream 32 filtre 3x3
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2), # reduce dimensiunea de 2 ori 16x16
    layers.Conv2D(64,(3,3), activation='relu', padding='same'), # 64 filtre 3x3
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2), # 8x8
    layers.Conv2D(128,(3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2), # 4x4
    layers.Flatten(),
    layers.Dense(512, activation='relu'), # Crestem la 512
    layers.Dropout(0.5), # 50% din imagini vor fi ignorate pentru antrenament mai eficient
    layers.Dense(100, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # obiect pt scadere lr
    initial_learning_rate=0.001, # lr de start
    decay_steps=500, # nr de pasi dupa care lr scade
    decay_rate=0.95 # factor de scadere, lr * decay_rate (0.95)
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs=50, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0) # testeaza modelul
print(f"Acuratete: {test_acc*100:.2f}%")
