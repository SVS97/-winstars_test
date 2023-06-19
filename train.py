import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import cv2
import random

from config import RANDOM_SEED, CSV_ANNOTATION, CORRUPTED_IMAGES, BATCH_SIZE, EPOCHS, checkpoint_filepath, \
    STEP_PER_EPOCH, TRAIN_DIR, IMG_SHAPE, NUM_CLASSES, VALIDATION_LENGTH, TEST_LENGTH, IMAGES_WITHOUT_SHIPS_NUMBER
from utils import calc_dices_all_image, rle_to_mask, one_hot
from model import Unet_model


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
    train_dataset = train_images.skip(VALIDATION_LENGTH + TEST_LENGTH)

    train_batches = (
        train_dataset
        .repeat()
        .batch(BATCH_SIZE))

    validation_batches = validation_dataset.batch(BATCH_SIZE)

    test_batches = test_dataset.batch(BATCH_SIZE)

    test_images = list(test_dataset.map(lambda x, y: x))
    test_labels = list(test_dataset.map(lambda x, y: y))

    # Set model
    model = Unet_model()
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')
    model.compile(optimizer='adam', loss='binary_crossentropy')
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='auto'
    )
    history = model.fit(train_batches, steps_per_epoch=STEP_PER_EPOCH, epochs=EPOCHS,
                        validation_data=validation_batches,
                        callbacks=[save_callback])
    model.load_weights(checkpoint_filepath)

    # metrics check
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(history.epoch, loss, 'r', label='Training loss')
    plt.plot(history.epoch, val_loss, 'C2', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()
    plt.savefig('losses.png')

    # dice validation
    scores_list = dict()
    threshold_list = [0.2]  # [x/100 for x in range(20,80,10)]

    for threshold in threshold_list:
        scores = []
        for i in range(len(test_images)):
            img = test_images[i][None, :, :, :]
            pred_mask = model.predict(img)
            dice = calc_dices_all_image(test_labels[i][:, :, 0], pred_mask, threshold=threshold)
            scores.append(dice)
        scores_list[threshold] = np.sum(scores) / (len(test_images))
    print('dice = ', scores_list)
