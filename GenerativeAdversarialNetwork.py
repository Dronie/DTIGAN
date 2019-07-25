#import keras
#from keras import layers
#from keras.preprocessing import image
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

# --------------------META DATA--------------------
latent_dim = 32
height = 32
width = 32
channels = 3

# --------------------GENERATOR NETWORK--------------------
generator_input = tf.keras.Input(shape=(latent_dim,))

# Transforms the input into a 16 x 16, 128-channel feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsamples to 32 x 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# produces a 32 x 32, 1-channel feature map (32 x 32 is the image size)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = tf.keras.models.Model(generator_input, x)
generator.summary()

# --------------------U-NET--------------------
def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.LeakyReLU()(encoder) # CHANGE
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.LeakyReLU()(encoder) # CHANGE
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.AveragePooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU()(decoder) # CHANGE
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU()(decoder) # CHANGE
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU()(decoder) # CHANGE
  return decoder

def decoder_block_end(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU()(decoder) # CHANGE
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU()(decoder) # CHANGE
  decoder = layers.Conv2D(channels, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = tf.keras.layers.Activation('tanh')(decoder) # CHANGE
  return decoder

uNet_input = tf.keras.Input(shape=(latent_dim,))
num_filters = 128
# Latent Space Dim
# 32,

# # Transforms the input into a 32 x 32, 128-channel feature map
x = layers.Dense(128 * 32 * 32)(uNet_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((32, 32, 128))(x)
# 32, 32, 128

# Encode 1
encoder_block_1, skip_con_1 = encoder_block(x, num_filters)
# out: 16, 16, 128 

# Encode 2
encoder_block_2, skip_con_2 = encoder_block(encoder_block_1, num_filters)
# out: 8, 8, 128

# Encode 3
encoder_block_3, skip_con_3 = encoder_block(encoder_block_2, num_filters)
# out: 4, 4, 128

# Encode 4
encoder_block_4, skip_con_4 = encoder_block(encoder_block_3, num_filters)
# out: 2, 2, 128


# Decode 1
decoder_block_1 = decoder_block(encoder_block_4, skip_con_4, num_filters)
# Conv2DTranspose Layer out: 2, 2, 128
# Concat: 4, 4, 256
# Conv2D (layer out): 2, 2, 128

# Decode 2
decoder_block_2 = decoder_block(decoder_block_1, skip_con_3, num_filters)
# Conv2DTranspose Layer out: 4, 4, 128
# Concat: 8, 8, 256
# Conv2D (layer out): 4, 4, 128

# Decode 3
decoder_block_3 = decoder_block(decoder_block_2, skip_con_2, num_filters)
# Conv2DTranspose Layer out: 8, 8, 128
# Concat: 16, 16, 265
# Conv2D (layer out): 8, 8, 128

# Decode 4
decoder_block_4 = decoder_block_end(decoder_block_3, skip_con_1, num_filters)
# Conv2DTranspose Layer out: 16, 16, 128
# Concat: 32, 32, 256
# Conv2D: 32, 32, 128
# Conv2D: 32, 32, 3

uNet = tf.keras.models.Model(uNet_input, decoder_block_4)
uNet.summary()

# --------------------DISCRIMINATOR NETWORK--------------------
discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# Dropout layer to add stohasticity
#x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)

discriminator = tf.keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = tf.keras.optimizers.SGD(lr=0.0008, momentum=0.1, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# --------------------AVERSARIAL NETWORK--------------------
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)

gan_optimizer = tf.keras.optimizers.Adam(lr = 0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# --------------------IMPLEMENTING GAN TRAINING--------------------
(x_train, y_train), (_,_) = tf.keras.datasets.cifar10.load_data()

# select frog images
x_train = x_train[y_train.flatten() == 6]

# normalize the data
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 200
batch_size = 20
#seccifies where to save images
save_dir = '.'

start = 0

a_losses = []
d_losses = []

for step in range(iterations):
    
    #samples random points (Gaussian Distribution) in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    generated_images = generator.predict(random_latent_vectors)
    
    # combines them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    # assembles labels, discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    # adds noise to the labels
    labels += 0.05 * np.random.random(labels.shape)
    
    d_loss =discriminator.train_on_batch(combined_images, labels)
    
    # samples random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    misleading_targets = np.zeros((batch_size, 1))
    
    # trains the generator(via the gan model, where the discriminator weights are frozen)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    
    # occasionall saves and plots
    a_losses.append(a_loss)
    d_losses.append(d_loss)

    print(step)

    if step % 10 == 0 and step != 0:
        gan.save_weights('gan.h5')

        plt.plot(range(step + 1), a_losses, 'b')
        plt.plot(range(step + 1), d_losses, 'r')
        plt.show()
        
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))











