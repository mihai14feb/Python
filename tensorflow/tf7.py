import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# incarc si preprocesez
(x_train,_), (_,_) = tf.keras.datasets.mnist.load_data() # 60k imagini antrenare, 10k test
x_train = x_train.astype('float32') / 255.0 # da pixelilor o valoare intre 0 si 1, va fi mai usor de invatat aceasta normalizare
x_train = x_train * 2 - 1 # ajustare pentru a fi intervalul [0,1], deoarece este folosit tanh
x_train = x_train.reshape(-1,28,28,1) # -1 inseamna dimensiunea initiala (60k), imaginea 28x28 si 1 canal (alb negru)

# construiesc generatorul, transforma zgomot aleatoriu intr-o imagine
def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=100, activation='relu'), # intrarea este un vector de zgomot de 100 dimensiuni
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28*28*1, activation='tanh'), # produce la iesire un vector de 28*28*1 dimensiuni, tanh produce valori in intervalul[-1,1]
        layers.Reshape((28,28,1)) # transforma vectorul 28*28*1 intr-o imagine 28x28 cu un canal alb negru
    ])

    return model

# construiesc discriminatorul
def make_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28,28,1)), # aplatizeaza imaginea intr-un vector de 784 dimensiuni
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid') # 0 img falsa, 1 imagine reala
    ])

    return model

# creez modelele
generator = make_generator()
discriminator = make_discriminator()

discriminator.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# construiesc GAN
discriminator.trainable = False # dezactivam antrenarea discriminatorului
gan_input = layers.Input(shape=(100,)) # defineste intrarea gan, un vector cu 100 dimensiuni
fake_image = generator(gan_input) # trece zgomotul in generator pentru a creea o imagine falsa
gan_output = discriminator(fake_image) # trece img falsa in discriminator pentru a genera o probabilitate
gan = tf.keras.Model(gan_input, gan_output) # creeaza modelul gan, zgomot -> generator -> imagine falsa -> discriminator -> probabilitate
gan.compile(optimizer='adam',
            loss='binary_crossentropy')

def train_gan(epochs=10000, batch_size=32): # 10000 cicluri, 32 img procesate simultan
    for epoch in range(epochs):
        # antreneaza discriminatorul
        real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)] # 0, 60k, 32, genereaza 32 indici aleatorii
        noise = np.random.normal(0,1, (batch_size,100)) # genereaza zgomot aleatoriu (0,1), forma 32,100 adica 100 dim
        fake_images = generator.predict(noise, verbose=0) # trece zgomotul in generator pt a crea imagini false
        x = np.concatenate([real_images, fake_images]) # pune imaginile reale si false intr-un singur array
        y = np.concatenate([np.ones((batch_size,1)), np.zeros((batch_size,1))]) # creeaza eitchete, 1 img reale, 0 img false
        d_loss = discriminator.train_on_batch(x,y) # antreneaza discriminatorul pe un batch

        # antreneaza generatorul
        noise = np.random.normal(0,1, (batch_size, 100)) # genereaza un nou batch de zgomot
        y = np.ones([batch_size,1]) # creeaza etichete de 1, sa creada discriminatorul ca imaginile sunt reale
        g_loss = gan.train_on_batch(noise, y) # antreneaza generatorul in prin gan, generator -> imagine falsa -> discriminator -> probabilitate

        # Afiseaza programul
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
            noise = np.random.normal(0,1,(16,100)) # genereaza zgomot pt 16 img (pt vizualizare)
            generated_images = generator.predict(noise, verbose=0) # genereaza 16 imagini false
            plt.figure(figsize=(4,4)) # creeaza o figura pt afisarea img
            for i in range(16):
                plt.subplot(4,4,i+1)
                generated_images = (generated_images + 1) / 2  # transforma [-1,1] in [0,1]
                plt.imshow(generated_images[i,:,:,0], cmap='gray') # afiseaza img, elimina dimensiunea canalului pt afisare, afiseaza in tonuri de gri
                plt.axis('off')
            plt.show()

# antreneaza gan
train_gan(epochs=10000, batch_size=32)