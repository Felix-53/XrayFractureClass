# Project Title: Xray Fracture Classification


## Introduction 

This project uses a machine learning pipeline to to catergorise fracture images from X-rays. This project uses a CNN (Convolutional Neural Network), and trains it to catergorise the bone X-rays into fractured and non fractured using a Kaggle X-ray Fracture classification database. The X-ray database is saved and split into three catergories: train (training), val (validation) and test (testing).
The ML Pipeline below shows the full process of this project, starting with the Data Collection and verification, to remove non-images and verify the data. This is followed by an EDA to allow for an understanding of the the data and the image sizeing and labels. Finaly the CNN models are produced with differing architecture to evalutate there effect on performace and accuracy with V2 producing a more accurate model. 
 This project trained and compared two CNN models comparing their outputs and suggesting future changes, limitations and issues the occured.


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

CNNS (Convulutional Neural Networks) are widely used for image classificaiton as they are effective at recognising patterns making them ideal for recognising fractures compared to non fractures.

### Model 1 Architecture

'''python
def create_tf_model_v1(): #Creates model v1
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))   

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

#model_v2 = create_tf_model_v2()
#model_v2.fit(train_set,epochs=25,steps_per_epoch=len(train_set.classes) // batch_size,validation_data=validation_set,callbacks=[early_stop],verbose=1)

#import tensorflow as tf
#model = tf.keras.models.load_model(f'{file_path_v1}/fracture_detection_model_v1.keras')
#print('Model loaded successfully')
'''

### Model 2 Architecture

'''python
def create_tf_model_v2(): #Creates model v2
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))    

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

    #model_v2 = create_tf_model_v2()
#model_v2.fit(train_set,epochs=8,steps_per_epoch=len(train_set.classes) // batch_size,validation_data=validation_set,callbacks=[early_stop],verbose=1)

#import tensorflow as tf
#model = tf.keras.models.load_model(f'{file_path_v2}/fracture_detection_model_v2.keras')
#print('Model loaded successfully')
'''

### Differences in v1 to v2

The filter sizes were increased form 64 to 128 from version 1 to version 2, giving more capacity to learn from the Xray images and idealy more accurate results.





## 4. Model Evaluation 




## 5. Prediction 



## Jupyter Notebook Structure 

The Jupyter notebook is split into four main files, the data collection file, the data visulisation file and the model evaluation file. 

Outputs: Contrains the v1 and v2 model outputs, and their conusion matricies.

DataCollection : Sets up the folders, removes non-images, validates the data.

DataVisulisation : EDA, scatterplot of image dimentions, labels bar chart and some images displayed to verify quality.

Models_Evaluation : Generates the data, model 1 and 2 trainingm testing, evalulation, confusion matricies and prediction.


Each of the notebooks contain cells going through each process step by step.

## Future Work 

- Training using a GPU to incrase model training speed

- Retraining using a larger dataset to increse accuracy 



## Libraries and Modules 

 This is a list of the librarys used in the project:

### Numpy
Numpy is a python library that allows for scientific computing and data analysis. It has been used in this project to find the mean of the image size dimentions in the EDA and to convert probabilitys into class labels.

### Pandas
Pandas is a python data manipulation and analysis libary, it has been used to store counts for the bar charts, store the model training history. 

### Matplotlib
Matplotlib is a python libary for static, animaated and interactive visulisations, it has been used in this project to produce the plots and graphs, show the sample images and plot data.

### Seaborn
Seaborn is a python data visulisation libary, it was used to produce the label bar charts and the heat maps for the matricies.

### Scikit-Learn
Scikit-Learn is a python library for machine learning and data analysis, it has been used for the model evaluations, and recalling scores and model accuracy.

### Joblib
Jonlib is a python library that provides light weight pipelining, this has been used for saving and loading image shapes and classes. It has also been used for the pkl files allowing them for later use.

### OS
OS is gihubs operating system thats built in to the software, it has been used in a range of uses such as verifiying files exsist and listing file and directory contents. 

### PIL 
PIL is a python image processing library, it has been used to allow the partialy truncated images to still partialy be usefull without crashing the software. 

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