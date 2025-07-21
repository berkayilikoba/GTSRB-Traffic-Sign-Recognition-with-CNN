import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from preprocess import preprocess_data, IMG_WIDTH, IMG_HEIGHT, CHANNELS

MODEL_SAVE_DIR = "models"
MODEL_FILE_NAME = "gtsrb_model_best.keras"

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.arange(len(np.unique(y_true))),
                yticklabels=np.arange(len(np.unique(y_true))))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

print("Starting GTSRB Model Prediction Process...")

_, _, X_test, y_test_one_hot, num_classes = preprocess_data()

print(f"Test Data Shape: {X_test.shape}")
print(f"Test Labels Shape (one-hot): {y_test_one_hot.shape}")
print(f"Number of classes: {num_classes}")

model_filepath = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)

if not os.path.exists(model_filepath):
    print(f"Error: Trained model not found at '{model_filepath}'")
    print("Please run 'gtsrb_train.py' first to train and save the model.")
else:
    print(f"\nLoading trained model from: {model_filepath}")
    try:
        loaded_model = load_model(model_filepath)
        loaded_model.summary()
        evaluate_model(loaded_model, X_test, y_test_one_hot)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model file might be corrupted or incompatible with your Keras version.")
