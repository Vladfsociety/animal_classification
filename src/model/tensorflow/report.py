import matplotlib.pyplot as plt


def save_metrics(
	history,
	epochs,
	accuracy_title,
    accuracy_file_name,
    loss_title,
    loss_file_name,
):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.title(accuracy_title)
    plt.savefig(accuracy_file_name)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.title(loss_title)
    plt.savefig(loss_file_name)