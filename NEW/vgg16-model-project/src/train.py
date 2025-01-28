import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_vgg16_model

# Load training data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Training function
def train_model(model, data, epochs, batch_size):
    # Extract features and labels from data
    X_train = data.drop(columns=['Val Loss', 'Val Accuracy', 'Precision', 'Recall', 'F1 Score', 'mAP']).values
    y_train = data['Train Accuracy'].values

    # Model checkpoint callback
    checkpoint = ModelCheckpoint('vgg16_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], validation_split=0.2)

if __name__ == "__main__":
    # Load metrics data
    metrics_data = load_data(os.path.join('..', 'data', 'metrics.csv'))

    # Build and compile the VGG16 model
    vgg16_model = build_vgg16_model()
    
    # Train the model
    train_model(vgg16_model, metrics_data, epochs=100, batch_size=32)