# Project Title: Xray Fracture Classification


## Introduction 

This project uses a machine learning pipeline to to catergorise fracture images from X-rays. This project uses a CNN (Convolutional Neural Network), and trains it to catergorise the bone X-rays into fractured and non fractured using a Kaggle X-ray Fracture classification database. The X-ray database is saved and split into three catergories: train (training), val (validation) and test (testing). This project trained and compared two CNN models comparing their outputs and suggesting future changes, limitations and issues the occured.


## Business Objectives

The purpose of this project is to produces two CNN for fracture classification from Xray Images with an accuracy over 70%, this can be used in a clinical setting to help with processing xray images of fractures and help in faster detection and therefore treatment. This project aims to catergorise the fractures into two catergories, fractured and non fractured, with an accuracy over 70% while minimising errors. 


# ML Pipeline 

This ML Pipeline is the workflow of the project that shows the building and training and testing of the CNN model, below are the stages of the pipeline:


## 1. Data Collection 

The dataset used in this project is Bone Fracture Binary Classification from kaggle. The first stage for the dataprocessing is the removal of any non-images or invalid image formats, this is done by using the code below 

'''python
import os
def remove_non_image_file(my_data_dir):
    image_extension = ('.png', '.jpg', '.jpeg') # Allowed image file options
    
    folders = os.listdir(my_data_dir) # List of all in the file
    for folder in folders: #Loops over all in the folders
        folder_path = my_data_dir + '/' + folder # Path to the file/item
        if not os.path.isdir(folder_path): #Checks the items not a directory
           

             continue
    
    
    files = os.listdir(folder_path)
    non_images = 0 #  to count non image files
    images = 0 #  to count image files
    for given_file in files: # loops over all in the folders
           
            if not given_file.lower().endswith(image_extension): # Checks the file is not an img
                file_location = os.path.join(my_data_dir, folder)
                os.remove(file_location) # Deletes the non image files
                number_of_non_images +=1
            else:
                number_of_images += 1

    print(f"The Folder: {folder} - has this many image files",{number_of_images})
    print(f"Folder: {folder} - has this many non-image", {number_of_non_images})
'''   

This uses a list of accepted image extentions and removes and deletes the non-images then counting the amount of valid and the non-images. This prevents the program not working due to invalid files. 

The dataset is then split into three catergories, train ,test and val. This isnt nessesarily requried as the selected database is alrady split so this is a verification. The data is split at the ratio of 70% to train, 20% to validation and 10% to testing.

'''python
import os 
print (os.listdir('../data/test/')) # Lists all in test folder

split_train_validation_test_images(my_data_dir=f"../data/test/", # Splits the images
train_set_ratio=0.7, # 70% of images go to training
validation_set_ratio=0.1, # 10% of images go to val
test_set_ratio=0.2 # 20% of images go to test
                                   )
'''

## 2. EDA 

An EDA (Exploratory data analysis) has been conducted in the data visulisation notebook, this has been conducted to allow for an  understanding of the data before the model was built.

An analysis on the images was run to allow for a standard image size to be input to the CNN, the images show a range of sizes with a scatterplot comparing the width and height created. The image size of 224,224 was selected as the standard size.

A barchart was produced to compare the amount of fractured to non fractured images, with a reasonable balance of the images verified and not requiring rebalancing.

Some of the images were displayed to verifiy their quality and labeling was correct.



## 3. Model Building 

Two CNN models were produced using TensorFlow/Keras, allowing for comparison of the models. 


## 4. Model Evaluation 




## 5. Prediction 



## Jupyter Notebook Structure 

The Jupyter notebook is split into three main files, the data collection file, the data visulisation file and the model evaluation file. 

DataCollection : Sets up the folders, removes non-images, validates the data.

DataVisulisation : EDA, scatterplot of image dimentions, labels bar chart and some images displayed to verify quality.

Models_Evaluation : Generates the data, model 1 and 2 trainingm testing, evalulation, confusion matricies and prediction.


Each of the notebooks contain cells going through each process step by step.

## Future Work 

- Training using a GPU to incrase model training speed

- Retraining using a larger dataset to increse accuracy 



## Libraries and Modules 

TensorFlow / Keras

Numpy

Pandas

Matplotlib

Seaborn

Scikit Learn

Joblib

OS

PIL 


## Unfixed Bugs 

Some of the images from the database are corrupted, leading to them only being able to be partialy used. 

The CNN are trained only on the CPU increasing the training time and slowing progress.


## Acknowledgements and References

Ai used to error check, suggest fixes/changes that were all review and checked and helped with evaluation and feedback 

Code Institute Malaria Walkthrough 
This code walkthrough has been used as basis for structure and large amounts of code have bben used ect 

Data Set 

UOP Module / Kaggle File 

## Conclusions 