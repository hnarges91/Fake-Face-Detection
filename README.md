# Exploring Feature Map Correlations for Effective Fake Face Detection


## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Results](#results)

## Introduction

This project aims to detect deepfake images using feature extraction from pre-trained models. The features are used to train a neural network classifier that can distinguish between real and fake images.

## Setup

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Usage
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Place your datasets in the appropriate directories as specified in the code.
3. Run the `Face_Fake_Detection` function specifying the pre-trained model, and dataset.
4. Evaluate the model's performance by examining the classification report, confusion matrix, and plots for accuracy and loss.
5. The results including accuracy and loss metrics will be displayed.

## Results

The results of the model training and evaluation will be printed in the console, including:

- Mean accuracy and loss
- Median accuracy and loss
- Classification report
- Confusion matrix
- Accuracy and loss plots

### Example Code Snippet
 ```bash
model_name = 'vgg16'
dataset_name = 'example_dataset'
pooling_type = 'correlation'

accuracy, loss = Face_Fake_Detection(dataset_name,feature_kind,model_name)
print("Accuracy:", accuracy)
print("Loss:", loss)
 ```
