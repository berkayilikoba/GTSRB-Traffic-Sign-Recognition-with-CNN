import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from preprocess import preprocess_data
from model import create_gtsrb_model

MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_FILE_NAME = "gtsrb_model_best.keras"

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("Starting GTSRB Model Training Process...")

# 1. Load and preprocess data
X_train_full, y_train_one_hot, _, _, num_classes = preprocess_data()

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_one_hot, test_size=0.2, random_state=42, stratify=y_train_one_hot
)

print(f"\nOriginal Training Data Shape: {X_train_full.shape}")
print(f"Training Set Shape: {X_train.shape}")
print(f"Validation Set Shape: {X_val.shape}")
print(f"Number of classes: {num_classes}")

# 2. Create the model
model = create_gtsrb_model(num_classes)
model.summary()

# 3. Train the model
model_filepath = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)

checkpoint = ModelCheckpoint(
    model_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, verbose=1, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1
)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks_list,
    verbose=1
)

print("Model training finished.")
print(f"Best model saved to: {model_filepath}")

plot_training_history(history)
