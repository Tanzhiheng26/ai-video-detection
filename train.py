#%%
from pathlib import Path
import keras
from keras import layers
import tensorflow as tf
from model import Conv2Plus1D, ResizeVideo, add_residual_block
from dataloader import FrameGenerator
#%%
n_frames = 20
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(Path("data/train"), n_frames, training=True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

# val_ds = tf.data.Dataset.from_generator(FrameGenerator(Path("data/val"), n_frames),
#                                         output_signature = output_signature)
# val_ds = val_ds.batch(batch_size)
#%%
# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(input, x)

model.compile(loss = keras.losses.BinaryCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 10)
#%%
model.save_weights("checkpoints/mixed_20frames_10epochs.weights.h5")
# %%
