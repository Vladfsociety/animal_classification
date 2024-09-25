import time
from tensorflow.keras import mixed_precision
from .build import VGG11
from ..load_data import get_train_test_datasets
from ..report import save_metrics


def train():
    epochs = 20

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_dataset, validation_dataset = get_train_test_datasets()

    model = VGG11()

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs
    )

    print("--- Model train time: %s seconds ---" % round(time.time() - start_time, 4))

    save_metrics(
        history,
        epochs,
        "Training and Validation Accuracy (Tensorflow VGG11 like model)",
        "reports/tensorflow/vgg11/accuracy.jpg",
        "Training and Validation Loss (Tensorflow VGG11 like model)",
        "reports/tensorflow/vgg11/loss.jpg"
    )

    model.save('models/tensorflow/vgg11.keras')

if __name__ == '__main__':
    train()