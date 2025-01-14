import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
file_path = "metrics.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Plotting function
def plot_all_metrics(data):
    epochs = data['Epoch']

    # Create a figure with subplots for each metric
    metrics = ['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy', 'Precision', 'Recall', 'F1 Score', 'mAP']
    plt.figure(figsize=(20, 15))

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(4, 2, i)
        plt.plot(epochs, data[metric], label=metric, marker='o')
        plt.title(f'{metric} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("metrics_over_epochs.png", dpi=300)
    plt.show()

# Call the plotting function
plot_all_metrics(data)
