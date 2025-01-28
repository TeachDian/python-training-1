import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.utils.data_preprocessing import load_data, preprocess_data
from src.models.vgg16 import VGG16Model
from src.models.resnet import ResNetModel
from src.models.pnn import PNNModel
from src.models.knn import KNNModel
from src.models.svm import SVMModel

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

def main(args):
    # Load and preprocess data
    X, y = load_data(args.data_dir)
    X = preprocess_data(X)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'vgg16': VGG16Model(),
        'resnet': ResNetModel(),
        'pnn': PNNModel(),
        'knn': KNNModel(),
        'svm': SVMModel()
    }

    # Train each model
    for model_name, model in models.items():
        print(f'Training {model_name}...')
        train_model(model, X_train, y_train, X_val, y_val)
        model.save(os.path.join(args.model_dir, f'{model_name}.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train machine learning models.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save trained models.')
    args = parser.parse_args()
    
    main(args)