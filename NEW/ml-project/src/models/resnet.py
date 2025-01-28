from tensorflow.keras import layers, models

def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    for filters in [64, 128, 256, 512]:
        for _ in range(2):
            shortcut = x
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, shortcut])
            x = layers.ReLU()(x)

        x = layers.MaxPooling2D((2, 2))(x)

    # Fully connected layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Example usage:
# model = build_resnet(input_shape=(224, 224, 3), num_classes=10)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])