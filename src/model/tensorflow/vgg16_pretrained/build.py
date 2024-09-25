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

def VGG16_pretrained():
    learning_rate = 0.0001
    img_height = 224
    img_width = 224
    num_classes = 10

    model = Sequential()
    model.add(layers.Input(shape=(img_height, img_width, 3))) 
    model.add(get_data_augmented_layer())
    model.add(layers.Rescaling(1. / 255))
    model.add(layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2]))

    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model.add(base_model)

    print(base_model.summary())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))  # Assume 10 classes

    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model