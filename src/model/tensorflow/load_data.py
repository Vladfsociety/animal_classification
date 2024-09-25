import tensorflow as tf


def get_train_test_datasets():
    batch_size = 64
    img_height = 224
    img_width = 224

    train_dataset, validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'data',
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset