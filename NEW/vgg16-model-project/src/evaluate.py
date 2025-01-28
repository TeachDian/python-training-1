import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from model import build_model  # Assuming build_model is a function in model.py that returns the VGG16 model

def load_metrics(file_path):
    metrics = pd.read_csv(file_path)
    return metrics

def evaluate_model(model, validation_data, validation_labels):
    predictions = model.predict(validation_data)
    predicted_classes = np.argmax(predictions, axis=1)

    precision = precision_score(validation_labels, predicted_classes, average='weighted')
    recall = recall_score(validation_labels, predicted_classes, average='weighted')
    f1 = f1_score(validation_labels, predicted_classes, average='weighted')
    mAP = average_precision_score(validation_labels, predictions, average='weighted')

    return precision, recall, f1, mAP

def main():
    metrics_file_path = '../data/metrics.csv'  # Adjust path as necessary
    metrics = load_metrics(metrics_file_path)

    model = build_model()  # Load or build your VGG16 model here
    # Load your validation data and labels here
    validation_data = None  # Placeholder for validation data
    validation_labels = None  # Placeholder for validation labels

    precision, recall, f1, mAP = evaluate_model(model, validation_data, validation_labels)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'mAP: {mAP}')

if __name__ == '__main__':
    main()