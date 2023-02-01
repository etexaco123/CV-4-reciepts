###############################################################
# This file contains packages for the tensorflow library 
# - used to preprocess the image files 
# - Uses Transfer learning on VGG16 CNN architecture for training and fine-tuning
# _ I experimented with different architecture from VGG 16, Mobilenet and ResNet which all have an input of 224 X 224
# - I also experimented with different loss function -Huber loss, MSE, and Intersection over Union but only MSE seems 

###############################################################
from utils import config
import tensorflow as tf
from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Dense, Conv2D , MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# load the contents of the CSV annotations file
print("[INFO] loading dataset...")
rows = open(config.ANNOTS_PATH).read().strip().split("\n")
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []

# loop over the rows
count =0
for row in rows:
	# break the row into the filename and bounding box coordinates
	row = row.split(",")
	(filename, startX, startY, endX, endY) = row
    # derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	startX = float(startX) / w
	startY = float(startY) / h
	endX = float(endX) / w
	endY = float(endY) / h

    # load the image and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	# update our list of data, targets, and filenames
	data.append(image)
	targets.append((startX, startY, endX, endY))
	filenames.append(filename)


# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

#####################################################################
# Design of intersection over union loss
# def IoU(y_true, y_pred, epsilon=1e-6):
#     # Flatten the predictions and true values
#     y_true_flatten = tf.reshape(y_true, [-1])
#     y_pred_flatten = tf.reshape(y_pred, [-1])
    
#     # Calculate the Intersection area
#     intersection = tf.reduce_sum(y_true_flatten * y_pred_flatten)
    
#     # Calculate the Union area
#     y_true_area = tf.reduce_sum(y_true_flatten)
#     y_pred_area = tf.reduce_sum(y_pred_flatten)
#     union = y_true_area + y_pred_area - intersection
    
#     # Avoid division by zero
#     iou = (intersection + epsilon) / (union + epsilon)
    
#     return iou

# def IoU_loss(y_true, y_pred):
#     iou = IoU(y_true, y_pred)
    
#     # Calculate the IoU loss
#     loss = -tf.math.log(iou)
    
#     return loss

#############################################################################################
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor =Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
# for layer in resnet.layers:
# 	vgg.trainable = False
vgg.trainable = False
# unfreezing the first 2 layers
# for layer in vgg.layers[:2]:
#     layer.trainable = True
#
# Loop over the layers
# for i, layer in enumerate(vgg.layers):
#     # Set the trainable attribute to True for the first 2 layers and the last 2 layers
#     if i < 2 or i >= len(vgg.layers) - 2:
#         layer.trainable = True
#     else:
#         layer.trainable = False
# flatten the max-pooling output of VGG

flatten = vgg.output #edited
flatten = Flatten()(flatten)
# convs = vgg.layers[-3:] # used to unfreez the last layer

# for conv in convs:
#     x = conv(x)
#################################################################################
# tried for covolution not giving very good results
# flatten = Flatten()(flatten) #edited
# flatten = Flatten()(x)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
# inputs = Input(shape=(224, 224, 3))

# # Add the first convolutional layer
# conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
# # Add the max pooling layer
# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# # Add the second convolutional layer
# conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
# # Add the max pooling layer
# pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# # Add the third convolutional layer
# conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool2)
# # Add the max pooling layer
# pool3 = MaxPooling2D(pool_size=(2, 2))(conv2)


# flatten = Flatten()(pool3)
# bboxHead = Dense(128, activation="relu")(flatten)
# bboxHead = Dense(64, activation="relu")(flatten)
# bboxHead = Dense(32, activation="relu")(bboxHead)
# bboxHead = Dense(4, activation="sigmoid")(bboxHead)

############################################################################

# bboxHead = Dense(256, activation="relu")(flatten)
# bboxHead = Dropout(0.2)(bboxHead)
bboxHead = Dense(256, activation="relu")(flatten)
bboxHead = Dropout(0.2)(bboxHead)
bboxHead = Dense(128, activation="relu")(bboxHead)
bboxHead = Dropout(0.2)(bboxHead)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dropout(0.2)(bboxHead)
bboxHead = Dense(4, activation="linear")(bboxHead)

# construct the model we will fine-tune for bounding box regression
# initialize the optimizer, compile the model, and show the model
# summary
model = Model(inputs=vgg.input, outputs=bboxHead)
opt = Adam(learning_rate=config.INIT_LR)
# opt = RMSprop(learning_rate=config.INIT_LR)
# model.compile(loss="mse", optimizer=opt)
# loss_fn = tf.keras.losses.Huber()
model.compile(loss="mse", optimizer=opt)
# model.compile(loss=IoU_loss, optimizer=opt)
# model.compile(loss=loss_fn, optimizer=opt)
print(model.summary())

# Create an EarlyStopping callback
# early_stop = EarlyStopping(monitor='val_loss', patience=10)

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1 ) #callbacks = [early_stop]

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
