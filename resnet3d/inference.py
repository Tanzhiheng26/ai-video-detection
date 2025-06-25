# %%
from pathlib import Path
import keras
from keras import layers
import tensorflow as tf
from model import Conv2Plus1D, ResizeVideo, add_residual_block
from dataloader import FrameGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# %%
HEIGHT = 224
WIDTH = 224

n_frames = 16
batch_size = 8

# %%
output_signature = (tf.TensorSpec(shape = (n_frames, HEIGHT, WIDTH, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

test_ds = tf.data.Dataset.from_generator(FrameGenerator(Path("../data/test"), n_frames),
                                          output_signature = output_signature)
# Batch the data
test_ds = test_ds.batch(batch_size)

# %%
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

# %%
model.load_weights("checkpoints/05.weights.h5")
model.compile(loss = keras.losses.BinaryCrossentropy(),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])

# %%
prob = model.predict(test_ds)
prob = tf.squeeze(prob)
y_pred = tf.cast(prob > 0.5, tf.int16)

# %%
y_true = [labels for _, labels in test_ds.unbatch()]
y_true = tf.stack(y_true)

# %%
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

# %%
acc = accuracy_score(y_true, y_pred)
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Accuracy:", acc)
print("Precision:", p)
print("Recall", r)
print("F1 score:", f1)
