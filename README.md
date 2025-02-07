# ASL Translator 🤟

**ASL Translator** is a machine learning-based application designed to recognize and translate American Sign Language (ASL) signs into text. Using a Convolutional Neural Network (CNN), this application detects ASL signs from images or real-time video and converts them into corresponding letters. It currently supports signs for the alphabet (A-Z), space, delete, and a "nothing" category. 

## Features

* **Custom Dataset Generation**: Users can create their own training and testing datasets using `generate_gesture_data.py`.
* **Model Training**: The `train_cnn.py` script supports training a CNN model using the Kaggle dataset (credits: https://www.kaggle.com/datasets/grassknoted/asl-alphabet) or user-generated images. 
* **Predefined Training and Testing Sets**: The project includes a `/train/` folder containing labeled images for training and a /test/ folder with images for testing. These images are categorized into different ASL signs to facilitate training and evaluation. 
* **Real-Time ASL Detection**: `gestures_model.py` enables real-time detection and translation of ASL signs. 

## Planned Improvements 

* Enhance the model's accuracy for better recognition. 
* Improve translation capabilities by displaying words instead of single letters when users change signs. 
* Develop a web-based version of the application for broader accessibility.

## Technologies Used

* **Python** for scripting and model training 
* **TensorFlow/Keras** for building and training the CNN model
* **OpenCV** for real-time image processing and gesture detection 
* **Numpy** for data handling and manipulation 
* **Matplotlib** for visualizing data and model performance 

## How to Use ASL Translator

#### 1. Generate Custom Training Data (Optional)
Run `generate_gesture_data.py` to capture custom images for training/testing. 

#### 2. Train the Model
Execute `train_cnn.py` to train the CNN using an existing ASL dataset or custom generated images using `generate_gesture_data.py` stored in `/train/`.

#### 3. Run Real-Time Detection 
Use `gestures_model.py` to detect ASL signs in real time and display the corresponding text output. 

## Setup Instructions 

Follow these steps to run ASL Translator locally: 

1. Clone the repository:

        git clone https://github.com/poojaiyengar123/ASL-Translator.git
        cd ASL-Translator

2. Do one of the following: 

    * Download an ASL dataset from Kaggle 
    * Generate your own training data

            python3 generate_gesture_data.py

3. Train the model:

        python3 train_cnn.py

4. Run real-time ASL detection: 

        python3 gestures_model.py