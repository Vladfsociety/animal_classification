import time
from tensorflow.keras import mixed_precision
from .build import CNN
from ..load_data import get_train_test_datasets
from ..report import save_metrics


def train():    
    epochs = 30

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    train_dataset, validation_dataset = get_train_test_datasets()

    model = CNN()

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
        "Training and Validation Accuracy (Tensorflow simple model)",
        "reports/tensorflow/simple_model/accuracy.jpg",
        "Training and Validation Loss (Tensorflow simple model)",
        "reports/tensorflow/simple_model/loss.jpg"
    )

    model.save('models/tensorflow/simple_model.keras')

if __name__ == '__main__':
    train()