import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# incarc dataset imdb, secvente de id-uri de cuvinte
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# preprocesare, taie sau completeaza secventele pentru a avea fix 300 cuvinte
max_length = 300
x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

# definirea modelului
model = models.Sequential([
    layers.Embedding(10000,128, input_length=max_length), # transforma id-urile de cuvinte in vectori de 128 dimensiuni
    layers.LSTM(128, return_sequences=True), # proceseaza secventa cuvintelor
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False), # false pentru a returna doar output la Dense
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)), # face greutatile mai mici pentru a optimiza l2 regularizer
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(1, activation='sigmoid')
])

scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=300,
    decay_rate=0.9
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, restore_best_weights=True) # daca loss nu scade timp de 2 epoci, opreste antrenarea
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1) # 20% validare, antreneaza modelul pe 80% din x_train, restul de 20% il testeaza pe date "nevazute"

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acuratete: {test_acc*100:0.2f}%")