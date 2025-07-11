import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Load and preprocess MNIST ---
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_cat = to_categorical(y_train, 10)

# --- Build model ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train model ---
print("Training model on MNIST...")
model.fit(x_train, y_train_cat, epochs=3, batch_size=64)

# --- Predict digits from folder ---
digit_counts = [0] * 10
folder = "/content/digits/digits"  # update path if needed

print("\nClassifying digits...")
for filename in tqdm(sorted(os.listdir(folder))):
    path = os.path.join(folder, filename)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img_resized = cv2.resize(img, (28, 28))
    img_norm = img_resized.astype('float32') / 255.0
    img_input = img_norm.reshape(1, 28, 28, 1)

    pred = np.argmax(model.predict(img_input, verbose=0))
    digit_counts[pred] += 1

# --- Output results ---
print("\n📊 Digit counts:", digit_counts)
print("🔢 Total images:", sum(digit_counts))
