
# Klippa ML Assesment

The project involves the use of computer vision algorithims to detect the total amount of money spent on groceries or various items puchased from a store as stated in the reciepts issued.


## Authors

- [@etexaco123](https://www.github.com/etexaco123)


## Problem Description
The problem identified for the project is clearly a bounging box regresion problem. This is because the 4 vertices of the ground truth box is clearly a continous variable. during inpection also noticed that some images are really blured or captured under poor lighting conditions and therefore requires a robust solution.

## Procedure  
- Created a virtual environment for the project 
- Run "pip install -r requirements.txt" at your project directory
- Download and extract the 500MB file from the email. Copy the **train** folder to the same directory as the output, config.
- Extract the bounding box information from the annotations.json  file to create the training and test datset.  
- To do this I created a function in the preprocess.py to help me convert the bounding box cordinates in the annotations file into pascal voc format which is sorted as xmin,ymin,xmax,ymax  
- To do this each files in the sub directory containing the **'document.jpg'**, **'annotations.json'** was called recusively and the ground truth bounding box information about the images in pascal voc format was appended to the file path of each image file in the subdirectories and written to CSV. 
- A ML model is designed and using transfer learning and fine-tuning.  
- The images are loaded, pixels are normalized and converted to arrays and reshaped to 224 X 224 input shape and fed to the model for training.  
- A test image or list of test images are fed into the model for bounding box prediction.  
- The predicted bounding box is used to crop the test image and fed the croped image to the Google Vision API by calling the rest API.  
- The total amount numbers is extracted from the croped image using the Vision API.  
- The folder directory from where the image was uploaded is returned as well as the respose from the the API

## Settings 
- **The file directory Settings**: this is implemented in the config file in the utils package I created. The project folder can be read from any current diectory you are in.  

- **CNN Training settings**: This also stored in the config file in the utils package this include the  
- learning rate = 1e - 5  
- number of epoch = 50
- batch size = 32

## Model Design and Architecture
Since the dataset is few I thought it was best to apply transfer learning technique to train on the pretrained weights and bias of deep neural net since we have a small dataset. To design the best solution I tried MobileNet, ResNet and VGG16, of all of them the VGG16 gave the most promising results on the training and validation. This CNN model was implemented in tensorflow. Also there was the challenge of selecting the best loss function for the task so I attempted different combination of methods among the MSE, Huber loss and intersection over Union(IOU) but I had to write a custom made function for IoU because we don't have a built-in function yet. But in the end the MSE performed better during warm-up steps.

![Alt](https://github.com/etexaco123/Klippa-ML-Assesment/blob/main/output/model_architecture.png)

## Proposed Method
- VGG16 Archtecture: I had to freeze earlier layers, warmed up the model and finetuned by unfreezing the trainable weights
- 3 layers of fully connected or dense layers was fitted on top
- There was a dropout of 20% in between each layers  
- The final output layer is fited only 4 neurons with sigmoid activation
- MSE loss function  
  
<!---![Alt](https://github.com/etexaco123/Klippa-ML-Assesment/blob/main/output/model_architecture.png) ---> 
Below is the plot for MSE loss on the traing and validation set
![Alt](https://github.com/etexaco123/Klippa-ML-Assesment/blob/main/output/plot.png)

## Training Strategy
The dataset was split at a 90% training and 10% testing set, I applied data augmentation but did not helpme. 

## Testing and Result  
When tested using the "test_image.jpg" or the "test_images.txt"  To run the program entering the command in the terminal  
$ "python predict.py --input output/test_image.jpg" or  

$ "python predict.py --input output/test_images.txt"
  
The resulting output image is shown in in the image below  
![Alt](https://github.com/etexaco123/Klippa-ML-Assesment/blob/main/output/output_image.png)

## OCR features
The OCR was implemented by using the REST API features to call the Google Vison API from your google cloud account. **Note** you have to have a google console account and enable billing and generate your API key which is in json format to use this feature. This can be found in the "ocr.py" file
- My file is intrested in only the description field of the API

## Working on extra features using Flask framework
The goal is to enable the images uploadable to the model to output the test image and its predicted bounding box 

To run the app you can use   
$ flask run

It should didplay the home page loading the index.html on the local host as shown  
  
![Alt](https://github.com/etexaco123/Klippa-ML-Assesment/blob/main/output/index_page_snapshot.png)  
  
## Directory structure
Vitual env  
train/ :contains the training images  
output/: contains the model in h5 format,, and other plots  
train.py  
predict.py  
preprocess.py  
ocr.py: used to call the Google VisionAPI ust RestAPI  
app.py is the file for loading the flask app
templates/
index.html  
predict.html 

