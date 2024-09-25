import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def get_data_augmented_layer():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ]
)

def VGG11():
    learning_rate = 0.0001
    img_height = 224
    img_width = 224
    num_classes = 10

    model = Sequential([
        layers.Input((img_height, img_width, 3)),
        get_data_augmented_layer(),
        layers.Rescaling(1. / 255),
        layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2]),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model