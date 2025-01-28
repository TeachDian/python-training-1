from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_vgg16_model(input_shape=(224, 224, 3), num_classes=10):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Create the model
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model