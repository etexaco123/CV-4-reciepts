# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
# Make sure this file is in the same diratory of your images folder 
abpath = os.getcwd()
BASE_PATH = abpath
IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, os.path.normpath(os.path.join('train', 'bb_info.csv'))])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-5
NUM_EPOCHS = 50
BATCH_SIZE = 32