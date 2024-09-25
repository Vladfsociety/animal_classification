import time
from tensorflow.keras import mixed_precision
from .build import VGG16_pretrained
from ..load_data import get_train_test_datasets
from ..report import save_metrics


def train():
    epochs = 10

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_dataset, validation_dataset = get_train_test_datasets()

    model = VGG16_pretrained()

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
        "Training and Validation Accuracy (Tensorflow VGG16 pretrained model)",
        "reports/tensorflow/vgg16_pretrained/accuracy.jpg",
        "Training and Validation Loss (Tensorflow VGG16 pretrained model)",
        "reports/tensorflow/vgg16_pretrained/loss.jpg"
    )

    model.save('models/tensorflow/vgg16_pretrained.keras')

if __name__ == '__main__':
    train()