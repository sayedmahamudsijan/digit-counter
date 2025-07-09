import os
import cv2
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Load sklearn's built-in digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Train a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Folder where digit images are stored
folder = "digits"
counts = [0] * 10  # For digits 0 through 9

# Process each image
for filename in tqdm(os.listdir(folder)):
    filepath = os.path.join(folder, filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Resize to 8x8 to match the training dataset format
    img_resized = cv2.resize(img, (8, 8))
    # Invert and scale to 0-16 range to match sklearn format
    img_scaled = 16 - (img_resized // 16)
    img_flattened = img_scaled.flatten().reshape(1, -1)

    # Predict digit
    pred = knn.predict(img_flattened)[0]
    counts[pred] += 1

# Output the result
print("Digit counts:", counts)
print("Total digits:", sum(counts))
