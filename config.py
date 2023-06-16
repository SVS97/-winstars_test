RANDOM_SEED = 77
# specify path to the dataset
TRAIN_DIR = '../airbus-ship-detection/train_v2/'
TEST_DIR = '../airbus-ship-detection/test_v2/'
CSV_ANNOTATION = '../airbus-ship-detection/train_ship_segmentations_v2.csv'
checkpoint_filepath = './checkpoints/model-checkpoint/'
CORRUPTED_IMAGES = '6384c3e78.jpg'

# Train parameters
VALIDATION_LENGTH = 2000
TEST_LENGTH = 2000
BATCH_SIZE = 2
BUFFER_SIZE = 1000
IMG_SIZE = 768
NUM_CLASSES = 2
EPOCHS = 50
STEP_PER_EPOCH = 100
