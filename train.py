import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, CSV_ANNOTATION, CORRUPTED_IMAGES, BATCH_SIZE, EPOCHS, checkpoint_filepath, \
    STEP_PER_EPOCH
from utils import area_isnull, calc_area_for_rle, calc_class, data_generator, create_data, calc_dices_all_image
from model import Unet_model

# set the random seed:
random.seed(RANDOM_SEED)

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

# split train-set and validation-set (stratified: area class)
train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
# split and make random train lists with ships and without it
train_isship_list = train['ImageId'][train['isnan'] == 0].tolist()
train_isship_list = random.sample(train_isship_list, len(train_isship_list))
train_nanship_list = train['ImageId'][train['isnan'] == 1].tolist()
train_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))

val_isship_list = train['ImageId'][train['isnan'] == 0].tolist()
val_isship_list = random.sample(train_isship_list, len(train_isship_list))
val_nanship_list = train['ImageId'][train['isnan'] == 1].tolist()
val_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))

# create data generator
CAP_NUM = min(len(train_isship_list), len(train_nanship_list))
datagen_train = data_generator(train_df, train_isship_list, train_nanship_list, batch_size=BATCH_SIZE, cap_num=CAP_NUM)
datagen_val = data_generator(train_df, val_isship_list, val_nanship_list, batch_size=BATCH_SIZE, cap_num=CAP_NUM)

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
history = model.fit(datagen_train, steps_per_epoch=STEP_PER_EPOCH, epochs=EPOCHS, callbacks=[save_callback])
model.load_weights(checkpoint_filepath)

# metrics check
loss = history.history['loss']
plt.figure()
plt.plot(history.epoch, loss, 'r', label='Training loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.savefig('losses.png')

# dice validation
val_list = val['ImageId'].tolist()
scores_list = dict()
threshold_list = [0.2]  # [x/100 for x in range(20,80,10)]
for threshold in threshold_list:
    scores = []
    for i in (range(len(val_list) // 2)):
        temp_list = val_list[i * 2:(i + 1) * 2]
        val_img, val_mask = create_data(train_df, temp_list, dice=True)
        pred_mask = model.predict(val_img)
        F2 = calc_dices_all_image(val_mask, pred_mask, threshold=threshold) * 2
        scores.append(F2)
    val_F2 = np.sum(scores) / (len(val_list) // 2 * 2)
    scores_list[threshold] = val_F2
print('scores_list = ', scores_list)
