import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import create_data, area_isnull, calc_area_for_rle, calc_class
from config import TRAIN_DIR, checkpoint_filepath, CSV_ANNOTATION, CORRUPTED_IMAGES, IMG_SIZE
from model import Unet_model
from imageio import imread
import pandas as pd


model = Unet_model()
model.load_weights(checkpoint_filepath)
opt_threshold = 0.2

# prepare images for inference
train_df = pd.read_csv(CSV_ANNOTATION)
# remove bug image
train_df = train_df[train_df['ImageId'] != CORRUPTED_IMAGES]

# remove 100000 non-ship images
train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
train_df['isnan'].value_counts()
train_df = train_df.sort_values('isnan', ascending=False)
train_df = train_df.iloc[100000:]
train_df['isnan'].value_counts()

# calculate ship area and group by ImageId
train_df['area'] = train_df['EncodedPixels'].apply(calc_area_for_rle)
# get small area of one ship; If estimated area of the ship is less than 10, it is corrected to 0.
train_df_isship = train_df[train_df['area'] > 0]
train_df_smallarea = train_df_isship['area'][train_df_isship['area'] < 10]
train_gp = train_df.groupby('ImageId').sum()
train_gp = train_gp.reset_index()

# set class of ship area
train_gp['class'] = train_gp['area'].apply(calc_class)
train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
val_list = val['ImageId'].tolist()


image_list = val_list[28:38]
fig, axes = plt.subplots(len(image_list), 3, figsize=(5,5*len(image_list)))
for i in range(len(image_list)):
    img = imread(TRAIN_DIR + image_list[i])
    input_img, gt_mask = create_data(train_df, [image_list[i]])
    pred_mask = model.predict(input_img)
    pred_mask = pred_mask > opt_threshold
    pred_mask = pred_mask.reshape(IMG_SIZE, IMG_SIZE, 1)
    gt_mask = gt_mask * 255
    gt_mask = gt_mask.reshape(IMG_SIZE, IMG_SIZE)
    pred_mask = pred_mask.reshape(IMG_SIZE, IMG_SIZE)
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(gt_mask)
    axes[i, 2].imshow(pred_mask)
plt.show()
