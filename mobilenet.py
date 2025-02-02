import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 50
learning_rate = 0.001
train_data_dir = 'path_to_train_data'  # Replace with your training data directory
val_data_dir = 'path_to_val_data'      # Replace with your validation data directory
output_dir = 'output'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# CSV Logger
csv_logger = CSVLogger(os.path.join(output_dir, 'metrics.csv'))

# Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[csv_logger]
)

# Save metrics plot
metrics_df = pd.read_csv(os.path.join(output_dir, 'metrics.csv'))

plt.figure(figsize=(20, 15))

# Plotting function
def plot_metrics(metrics_df):
    epochs = metrics_df['epoch']

    metrics = ['accuracy', 'val_accuracy', 'precision', 'val_precision', 'recall', 'val_recall']
    plt.figure(figsize=(20, 15))

    for i, metric in enumerate(metrics, start=1):
        plt.subplot(3, 2, i)
        plt.plot(epochs, metrics_df[metric], label=metric, marker='o')
        plt.title(f'{metric} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))

plot_metrics(metrics_df)