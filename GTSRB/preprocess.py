# gtsrb_preprocess.py
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

GTSRB_MAIN_DIR = "./GTSRB"
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 30, 30, 3

def load_data(data_type="train"):
    if data_type == "train":
        csv_file = os.path.join(GTSRB_MAIN_DIR, "Train.csv")
    elif data_type == "test":
        csv_file = os.path.join(GTSRB_MAIN_DIR, "Test.csv")
    else:
        raise ValueError("data_type must be 'train' or 'test'.")

    df = pd.read_csv(csv_file)
    images, labels = [], []

    for idx, row in df.iterrows():
        img_path = os.path.join(GTSRB_MAIN_DIR, row['Path'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        images.append(img)
        labels.append(row['ClassId'])

        if idx % 5000 == 0:
            print(f"{data_type} data: Processed {idx} images...")

    return np.array(images), np.array(labels)

def preprocess_data():
    print("Loading training data...")
    X_train, y_train_raw = load_data("train")
    print(f"Loaded {len(X_train)} training images.")

    print("Loading test data...")
    X_test, y_test_raw = load_data("test")
    print(f"Loaded {len(X_test)} test images.")

    num_classes = len(np.unique(np.concatenate([y_train_raw, y_test_raw])))
    print(f"Number of classes: {num_classes}")

    y_train = to_categorical(y_train_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)

    return X_train, y_train, X_test, y_test, num_classes

X_train, y_train, X_test, y_test, num_classes = preprocess_data()

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape (one-hot): {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape (one-hot): {y_test.shape}")
print(f"Number of classes: {num_classes}")

plt.imshow(X_train[0])
plt.title(f"Sample Image (Class: {np.argmax(y_train[0])})")
plt.axis('off')
plt.show()
