# GTSRB-Traffic-Sign-Recognition-with-CNN
Traffic sign classification using a convolutional neural network (CNN) trained on the GTSRB dataset for accurate road sign recognition.
This project implements a Convolutional Neural Network (CNN) model to recognize traffic signs from the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The project includes data preprocessing, model creation, training, and evaluation.


## Overview

The goal of this project is to build a robust CNN to classify traffic signs accurately using the GTSRB dataset. This dataset contains more than 50,000 images of traffic signs classified into 43 different categories.

---

## Dataset

The **GTSRB dataset** is publicly available and can be downloaded from [Kaggle GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb).

Make sure to organize the dataset folder as:

GTSRB/
├── Train.csv
├── Test.csv
├── Train/
│ ├── 00000/
│ ├── 00001/
│ └── ...
├── Test/
│ └── ...
├── preprocess.py # Data loading and preprocessing
├── model.py # CNN model creation
├── train.py # Training script
├── predict.py # Model evaluation and prediction script
├── models/ # Saved trained models
├── README.md # This file
└── GTSRB/ # Dataset folder (not included in repo)

## Installation

1. Clone this repository:

```bash
git clone [https://github.com/berkayilikoba/gtsrb-traffic-sign-recognition.git](https://github.com/berkayilikoba/gtsrb-traffic-sign-recognition.git)
cd gtsrb-traffic-sign-recognition
```

2. Create a virtual environment (optional but recommended):
3. 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
4. 
```bash
pip install -r requirements.txt
```

## Usage 
Data Preprocessing
Make sure the GTSRB dataset is in the correct directory (./GTSRB).

Run the preprocessing script to load and prepare the data:

```bash
python preprocess.py
```
This script loads images, resizes them, normalizes pixel values, and one-hot encodes labels.

Training the Model
Run the training script:
```bash
python train.py
```
Load and preprocess the data
Split the training data into training and validation sets
Create the CNN model
Train the model with callbacks for early stopping and learning rate reduction
Save the best model to models/gtsrb_model_best.keras
Plot training accuracy and loss graphs

Evaluating the Model
After training, evaluate the model on the test set:
```bash
python predict.py
```

This will:
Load the saved model
Evaluate accuracy and loss on test data
Show a classification report
Plot a confusion matrix heatmap

Model Architecture
3 convolutional layers with increasing filter sizes (32, 64, 128)
Batch normalization and max pooling after each conv layer
Dropout layers to reduce overfitting
Fully connected dense layer with 256 units
Output layer with softmax activation for 43 classes
Optimized using Adam optimizer with learning rate 0.001

## Results
The model achieves competitive accuracy on the GTSRB test dataset. Training and validation accuracy/loss graphs are saved during training. The confusion matrix helps visualize classification performance.

## License
This project is licensed under the MIT License.



