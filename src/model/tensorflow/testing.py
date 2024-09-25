import numpy as np
from tensorflow.keras.preprocessing import image


def get_pred_class(pred_value):
    pred_value = np.argmax(pred_value, axis=1)
    mapping = {0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep', 8: 'spider', 9: 'squirrel'}
    return mapping[pred_value[0]]

def image_test(model, image_path, f):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)

    actual_class = image_path.split('/')[1].split('_')[0]
    predicted_class = get_pred_class(predictions)

    print(f"Actual class: {actual_class}, Predicted class: {predicted_class}", file=f)

def run_test(model, key):
    with open(f'reports/tensorflow/{key}/test_result.txt', 'w') as f:
        print(f"Model: {key}", file=f)
        test_images = [
            "test/butterfly_test.jpeg",
            "test/butterfly_test_2.jpg",
            "test/cat_test.jpeg",
            "test/cat_test_2.jpg",
            "test/chicken_test.jpeg",
            "test/chicken_test_2.jpg",
            "test/cow_test.jpg",
            "test/cow_test_2.jpg",
            "test/dog_test.jpeg",
            "test/dog_test_2.jpg",
            "test/elephant_test.jpeg",
            "test/elephant_test_2.jpeg",
            "test/horse_test.jpg",
            "test/horse_test_2.jpg",
            "test/sheep_test.jpg",
            "test/sheep_test_2.jpg",
            "test/spider_test.jpg",
            "test/spider_test_2.jpg",
            "test/squirrel_test.jpeg",
            "test/squirrel_test_2.jpeg",
        ]
        for test_image in test_images:
            image_test(model, test_image, f)
