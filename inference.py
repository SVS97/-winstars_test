import matplotlib.pyplot as plt

from utils import rle_to_mask, one_hot
from config import TRAIN_DIR, checkpoint_filepath, CSV_ANNOTATION, CORRUPTED_IMAGES, RANDOM_SEED, \
    IMAGES_WITHOUT_SHIPS_NUMBER, \
    VALIDATION_LENGTH, TEST_LENGTH, BATCH_SIZE, IMG_SHAPE, NUM_CLASSES
from model import Unet_model
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import cv2


def predict(image):
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0].argmax(axis=-1)
    return pred_mask


def load_train_image(tensor) -> tuple:
    path = tf.get_static_value(tensor).decode("utf-8")

    image_id = path.split('/')[-1]
    input_image = cv2.imread(path)
    input_image = tf.image.resize(input_image, IMG_SHAPE)
    input_image = tf.cast(input_image, tf.float32) / 255.0

    encoded_mask = image_segmentation[image_segmentation['ImageId'] == image_id].iloc[0]['EncodedPixels']
    input_mask = np.zeros(IMG_SHAPE + (1,), dtype=np.int8)
    if not pd.isna(encoded_mask):
        input_mask = rle_to_mask(encoded_mask)
        input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
        input_mask = np.expand_dims(input_mask, axis=2)
    one_hot_segmentation_mask = one_hot(input_mask, NUM_CLASSES)
    input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

    class_weights = tf.constant([0.0005, 0.9995], tf.float32)
    sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32), name='cast_sample_weights')

    return input_image, input_mask_tensor, sample_weights


if __name__ == '__main__':
    # set number of images
    N = 5
    model = Unet_model()
    model.load_weights(checkpoint_filepath)

    # set the random seed:
    random.seed(RANDOM_SEED)

    df = pd.read_csv(CSV_ANNOTATION)
    df['EncodedPixels'] = df['EncodedPixels'].astype('string')

    # Delete corrupted images
    CORRUPTED_IMAGES = ['6384c3e78.jpg']
    df = df.drop(df[df['ImageId'].isin(CORRUPTED_IMAGES)].index)

    # Dataframe that contains the segmentation for each ship in the image.
    instance_segmentation = df

    # Dataframe that contains the segmentation of all ships in the image.
    image_segmentation = df.groupby(by=['ImageId'])['EncodedPixels'].apply(
        lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()

    # reduce the number of images without ships
    images_without_ships = image_segmentation[image_segmentation['EncodedPixels'].isna()]['ImageId'].values[
                           :IMAGES_WITHOUT_SHIPS_NUMBER]
    images_with_ships = image_segmentation[image_segmentation['EncodedPixels'].notna()]['ImageId'].values
    images_list = np.append(images_without_ships, images_with_ships)

    # remove corrupted images
    images_list = np.array(list(filter(lambda x: x not in CORRUPTED_IMAGES, images_list)))

    images_list = tf.data.Dataset.list_files([f'{TRAIN_DIR}{name}' for name in images_list], shuffle=True)
    train_images = images_list.map(lambda x: tf.py_function(load_train_image, [x], [tf.float32, tf.float32]),
                                   num_parallel_calls=tf.data.AUTOTUNE)

    validation_dataset = train_images.take(VALIDATION_LENGTH)
    test_dataset = train_images.skip(VALIDATION_LENGTH).take(TEST_LENGTH)

    f, ax = plt.subplots(N, 3, figsize=(10, 4 * N))
    i = 0
    for image, mask in test_dataset.take(N):
        mask = mask.numpy().argmax(axis=-1)
        ax[i, 0].imshow(image)
        ax[i, 0].set_title('image')
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('true mask')

        pred_mask = predict(image)
        ax[i, 2].imshow(pred_mask)
        ax[i, 2].set_title('predicted mask')
        i += 1

    plt.show()
