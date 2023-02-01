################################################
# This program preprocesses the reciepts images##
#################################################
# Makes a copy of the imagesand 
# crops the images to size to test the ground truth bounding boxes
# converts to gray scale
# reads the json file
# converts the annotation files from json to csv file for easy loading to the VGG16 CNN network
# used to extract the bouding box and convert it to pascal voc formart which is xmin, ymin, xmax, ymax
# The new csv file is called bb_info.csv in the train folder
# resize the images
# saves in the folder
# function to augment the images


import cv2
import os
import shutil
import json
import ast
from PIL import Image
import csv

# print(os.getcwd())
os.chdir(os.getcwd() + '\\train')
path = os.getcwd()

# print(path)

def extract_bbox(root_dir):
    with open(root_dir, 'r') as json_file:
        data = json.load(json_file)
        # Extract bounding box
        info = (str(data).strip('[]'))
        info = ast.literal_eval(info)
        tleft,_,bright,_ = tuple(info['points'])

    return (tleft[0],tleft[1],bright[0],bright[1])


# def copy_specific_files(dir_path, file_names):
#     count = 0
#     if os.path.exists(os.path.join(dir_path, "new_dir")):
#         print("The files have already been copied.")
#         return
#     for root, dirs, files in os.walk(dir_path):
#         new_sub_dir_path = os.path.join(root, "new_dir")
#         if not os.path.exists(new_sub_dir_path):
#             os.makedirs(new_sub_dir_path)
#         for file in files:
#             if file in file_names:
#                 file_path = os.path.join(root, file)
#                 shutil.copy2(file_path, new_sub_dir_path)

# Example usage
#copy_specific_files(path, ["document.jpg", "annotations.json"])


# def img_crop(path):
#     for subdir, dirs, files in os.walk(path):
#         for dir in dirs:
#             for subdir, dirs, files in os.walk(os.path.join(subdir, dir)):
#                 for file in files:
#                     # Check if the file is an image
#                     if file.endswith('.jpg'):
#                         # Open the image
#                         im = Image.open(os.path.join(subdir, file)).convert('L')

#                         # Set the bounding box coordinates (x1, y1, x2, y2)
#                         x1,y1,x2,y2= tuple(extract_bbox(os.path.join(subdir, 'annotations.json')))

#                         # Crop the image to the bounding box
#                         cropped_im = im.crop((x1,y1,x2,y2))

#                         # Save the cropped image in the same subdirectory
#                         cropped_im.save(os.path.join(subdir, "cropped_" + file))

# #img_crop(path)

#############################################################################################
def get_bb_info(path):
    with open('bb_info_v2.csv', mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        count = 0
        for subdir, dirs, files in os.walk(path):  
            for file in files:
                if file.endswith('.jpg'):
                    
                    img_path = os.path.join(subdir, file)
                    file_path = os.path.normpath(img_path)
                    dirname = os.path.dirname(file_path)
                    anno_path = os.path.normpath(os.path.join(dirname, 'annotations.json'))
                    # print(anno_path)
                    # x1,y1,x2,y2 = extract_bbox(anno_path)
                    x1,y1,x2,y2 = extract_bbox(anno_path)
                    img_parts = file_path.split(os.path.sep)
                    # dirn = os.path.dirname(img_path)
                    result = os.path.join(img_parts[-2], img_parts[-1])
                    dirn = os.path.normpath(result)
                    
                    writer.writerow([dirn, x1, y1,x2,y2])


# get_bb_info(path)
    ############################
        #             count+=1
        # print(count)


                    # dirname = os.path.dirname(file_path)
                    # basename = os.path.dirname(img_path)
                    # x1,y1,x2,y2= tuple(extract_bbox(os.path.join(subdir, 'annotations.json')))
                    # print(img_path,x1,y1,x2,y2)
                    # if os.path.exists('bb_info.csv'):
                    #     print("The files have already exist")
                    #     return
                    
                        # Add new rows to the CSV file
                    # writer.writerow([dirn, x1, y1,x2,y2])
                    # print(dirn)
                    # count+=1
                    # print(count)

# get_bb_info(path)




# # with open(csv_file, mode='w', newline='') as file:
# #         writer = csv.writer(file)
# #         writer.writerow(["File Path"])
# #         for root, dirs, files in os.walk(dir_path):
# #             for file in files:
# #                 file_path = os.path.join(root, file)
# #                 writer.writerow([file_path])


def read_json(x):
    # function to read json file
    with open(x) as json_file:
        data = json.load(json_file)
        info = (str(data).strip('[]'))
        info = ast.literal_eval(info)
        tleft,_,bright,_ = tuple(info['points'])
    return (tleft[0],tleft[1],bright[0],bright[1])

# root_dir = '/path/to/json/files'
# with open('bb_info.csv', mode='w', newline='') as file:
#         # Create a CSV writer object
#         writer = csv.writer(file)
#         for subdir, dirs, files in os.walk(path):
#             for file in files:
#                 if file.endswith("annotations.json"):
#                     filepath = os.path.join(subdir, file)
#                     x1,y1,x2,y2 = tuple(read_json(filepath))
#                     dirn = os.path.dirname(filepath)
#                     img_paths = os.path.join(dirn, 'document.jpg')
#                     img_parts = img_paths.split(os.path.sep)
#                     result = os.path.join(img_parts[-2], img_parts[-1])
#                     imagdir = os.path.normpath(result)
                    
#                     writer.writerow([imagdir, x1, y1,x2,y2])
            



###########################################
#Perform data augmentation
import albumentations as A
import pandas as pd
###########################################
#extract the labels from the value
def extract_json(x):
    # function to read json file
    with open(x) as json_file:
        data = json.load(json_file)
        info = (str(data).strip('[]'))
        info = ast.literal_eval(info)
        val = tuple(info['value'])
    return val



############################################
#My custom Augmentation
def custom_augmentation(image, val):
    # Define the augmentations to be applied
    augmentations = A.Compose([
        # A.HorizontalFlip(p=1),
                    #  A.Rotate(limit=90, p=1),
                     A.RandomContrast(limit=0.5, p=1),
                     A.RandomContrast(limit=0.9, p=1),
                     A.Blur(p=1, blur_limit=3),
                     A.Blur(p=1, blur_limit=5)
                    #  A.RandomCrop(height=100, width=100, p=1)
                     ], bbox_params={'format': 'pascal_voc', 'label_fields': list[x1,y1,x2,y2]})

    augmented_images = []
    for i in range(4):
        # Apply the augmentations to the image
        augmented_image = augmentations[i](image=image)['image']
        augmented_images.append(augmented_image)

    return augmented_images


# for subdir, dirs, files in os.walk(path):
#             for file in files:
#                 if file.endswith("annotations.json"):
#                     filepath = os.path.join(subdir, file)
#                     x1,y1,x2,y2 = tuple(read_json(filepath))
#                     dirn = os.path.dirname(filepath)
#                     img_paths = os.path.join(dirn, 'document.jpg')
###########################################
# b = 0
# for dir, dirname, files in os.walk(path):
#     for file in files:
#         if file.endswith('annotations.json'):
#             anno_path = os.path.join(dir, file)
#             x1,y1,x2,y2 = tuple(read_json(anno_path))
#             img_pat = os.path.join(dir, 'document.jpg')
#             # dir_path = os.path.dirname(img_pat)
#             # extract labels
#             # bname = img_path.split(os.path.sep)
#             # anno_path = os.path.join(dir, 'annotations.json')
            
#             # Load each original image in the folder
#             imag = cv2.imread(img_pat)
#             # Apply the custom augmentation function
#             augmented_images = custom_augmentation(imag, [x1,y1,x2,y2])
#             # Save each augmented image to a separate file
#             for i, augmented_image in enumerate(augmented_images):
#                 cv2.imwrite(str(dir) +"/augmented_image_{}.jpg".format(i), augmented_image)


#             # print(dir_path)
#             b+=1
# print(b)

from pathlib import Path
with open('bb_info_v2.csv', mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg"):
                    filepath = os.path.join(subdir, file)
                    img_parts = filepath.split(os.path.sep)
                    result = os.path.join(img_parts[-2], img_parts[-1])
                    imagdir = os.path.normpath(result)
                    x_path = Path("./" +str(img_parts[-2]) +"/annotations.json")
                    if x_path.is_file():
                        x1,y1,x2,y2 = read_json( x_path)
                        writer.writerow([imagdir, x1, y1,x2,y2])

                        # print([imagdir, x1, y1,x2,y2])
                        # x1,y1,x2,y2 = tuple(read_json(filepath))
                        # if file.endswith(".jpg"):
                        #     dirn = os.path.dirname(filepath)
                        #     imag_paths = os.path.join(dirn, file)
                        #     img_parts = imag_paths.split(os.path.sep)
                        #     result = os.path.join(img_parts[-2], img_parts[-1])
                        #     imagdir = os.path.normpath(result)

                        #     print([imagdir, x1, y1,x2,y2])
                
                        # writer.writerow([imagdir, x1, y1,x2,y2])
