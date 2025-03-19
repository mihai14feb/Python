import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

(x_train,_), (_,_) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train * 2 - 1
x_train = x_train.reshape(-1,28,28,1)

def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*128, input_dim=100),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7,7,128)),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh')
    ])

    return model

def make_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(28,28,1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

generator = make_generator()
discriminator = make_discriminator()
discriminator.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam',
            loss='binary_crossentropy')

def train_gan(epochs=15000, batch_size=64):
    for epoch in range(epochs):
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size,100))
        fake_images = generator.predict(noise, verbose=0)
        x = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size,1)), np.zeros((batch_size,1))])
        d_loss = discriminator.train_on_batch(x,y)

        noise = np.random.normal(0,1, (batch_size,100))
        y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise,y)

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            noise = np.random.normal(0,1,(16,100))
            generated_images = generator.predict(noise, verbose=0)
            generated_images = (generated_images + 1) / 2
            plt.figure(figsize=(4,4))
            for i in range(16):
                plt.subplot(4,4,i+1)
                plt.imshow(generated_images[i,:,:,0], cmap='gray')
                plt.axis('off')
            plt.show()

train_gan(epochs=15000, batch_size=64)