import numpy as np
import tensorflow as tf
from imageio import imread
from skimage.morphology import label
from config import IMG_SIZE, TRAIN_DIR


def area_isnull(x):
    if x == x:
        return 0
    else:
        return 1


def rle_to_mask(rle_list, SHAPE):
    """Change rle to mask. Returns mask."""
    tmp_flat = np.zeros(SHAPE[0] * SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i, v in zip(strt, length):
            tmp_flat[(int(i) - 1):(int(i) - 1) + int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask


def calc_area_for_rle(rle_str):
    """Calculates area for rle. Returns area."""
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area


def calc_class(area):
    """Sets class of ship area for each photo. Return number of class."""
    area = area / (IMG_SIZE * IMG_SIZE)
    if area == 0:
        return 0
    elif area < 0.005:
        return 1
    elif area < 0.015:
        return 2
    elif area < 0.025:
        return 3
    elif area < 0.035:
        return 4
    elif area < 0.045:
        return 5
    else:
        return 6


def data_generator(train_df, isship_list, nanship_list, batch_size, cap_num):
    """Make the ratio of is-ship images and nan-ship images  equal."""
    train_img_names_nanship = isship_list[:cap_num]
    train_img_names_isship = nanship_list[:cap_num]
    k = 0
    while True:
        if k + batch_size // 2 >= cap_num:
            k = 0
        batch_img_names_nan = train_img_names_nanship[k:k + batch_size // 2]
        batch_img_names_is = train_img_names_isship[k:k + batch_size // 2]
        batch_img = []
        batch_mask = []
        for name in batch_img_names_nan:
            tmp_img = imread(TRAIN_DIR + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((IMG_SIZE, IMG_SIZE, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (IMG_SIZE, IMG_SIZE))
                one_mask[:, :, 0] += tmp_mask
            batch_mask.append(one_mask)
        for name in batch_img_names_is:
            tmp_img = imread(TRAIN_DIR + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((IMG_SIZE, IMG_SIZE, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (IMG_SIZE, IMG_SIZE))
                one_mask[:, :, 0] += tmp_mask
            batch_mask.append(one_mask)
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)
        img = img / 255.0
        mask = mask / 255.0
        k += batch_size // 2
        yield img, mask


def dice_coefficient(y_true, y_pred, smooth=1):
    """Calculate Dice coefficient"""
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def calc_dices_one_image(mask_true, mask_pred):
    """Calculation score of 1 image. Returns score."""
    mask_true = mask_true.reshape(768, 768)
    mask_pred = mask_pred.reshape(768, 768)
    if mask_true.sum() == 0 and mask_pred.sum() == 0:
        dice = 1
    elif mask_true.sum() == 0 and mask_pred.sum() != 0:
        dice = 0
    elif mask_true.sum() != 0 and mask_pred.sum() == 0:
        dice = 0
    else:
        mask_label_true = label(mask_true)
        mask_label_pred = label(mask_pred)
        dice = dice_coefficient(mask_label_true, mask_label_pred)
    return dice


def calc_dices_all_image(batch_mask_true, batch_mask_pred, threshold=0.5):
    """Calculation score of all images. Returns score."""
    num = batch_mask_true.shape[0]
    tmp = batch_mask_pred > threshold
    batch_mask_pred = tmp.astype('int')
    dices = list()
    for i in range(num):
        dice = calc_dices_one_image(batch_mask_true[i], batch_mask_pred[i])
        dices.append(dice)
    return np.mean(dices)


def create_data(train_df, image_list, dice=False):
    """Creates data for testing. Returns image and mask for it"""
    batch_img = []
    batch_mask = []
    for name in image_list:
        tmp_img = imread(TRAIN_DIR + name)
        batch_img.append(tmp_img)
        mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
        one_mask = np.zeros((768, 768, 1))
        for item in mask_list:
            rle_list = str(item).split()
            tmp_mask = rle_to_mask(rle_list, (768, 768))
            one_mask[:, :, 0] += tmp_mask
        batch_mask.append(one_mask)
    img = np.stack(batch_img, axis=0)
    mask = np.stack(batch_mask, axis=0)
    img = img / 255.0
    if dice:
        mask = mask / 255.0
    return img, mask
