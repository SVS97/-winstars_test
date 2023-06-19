import numpy as np
import tensorflow as tf
from skimage.morphology import label
from config import IMG_SHAPE


def rle_to_mask(rle: str, shape=IMG_SHAPE):
    """
    :param rle: run length encoded pixels as string formated
           shape: (height,width) of array to return
    :return: numpy 2D array, 1 - mask, 0 - background
    """
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def dice_coefficient(y_true, y_pred, smooth=1):
    """Calculate Dice coefficient"""
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def calc_dices_one_image(mask_true, mask_pred):
    """Calculation score of 1 image. Returns score."""
    mask_pred = mask_pred.reshape(768, 768)
    mask_label_true = label(mask_true)
    mask_label_pred = label(mask_pred)
    dice = dice_coefficient(mask_label_true, mask_label_pred)
    return dice


def calc_dices_all_image(batch_mask_true, batch_mask_pred, threshold=0.5):
    """Calculation score of all images. Returns score."""
    # num = batch_mask_true.shape[0]
    tmp = batch_mask_pred > threshold
    batch_mask_pred = tmp.astype('int')
    dices = list()
    # for i in range(num):
    dice = calc_dices_one_image(batch_mask_true, batch_mask_pred)
    dices.append(dice)
    return np.mean(dices)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])
