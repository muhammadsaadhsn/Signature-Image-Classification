# Signature Classification and Detection Pipeline

## Overview

This project is a complete pipeline for detecting and classifying signature images. It processes raw images, extracts features, trains a convolutional neural network (CNN) classifier, and evaluates its performance. Additionally, it incorporates traditional feature extraction techniques (HOG and SIFT) as an optional component for experimentation.

---

## Directory Structure

```
├── Data                       # Raw images directory
├── Processed_Images           # Directory for bounding box-cropped images
├── Processed_Images_Train     # Training dataset directory
├── Processed_Images_Val       # Validation dataset directory
├── Processed_Images_Test      # Testing dataset directory
├── signature_classification_model.h5  # Trained CNN model
├── main_script.py             # Main script for the pipeline
├── README.md                  # Project documentation (this file)
```

---

## Steps in the Pipeline

### 1. Preprocessing and Bounding Box Detection

The raw images are processed to detect and crop the bounding boxes of the signatures.

#### Key Steps:
- Convert images to grayscale.
- Denoise using a median blur.
- Threshold the image to create a binary representation.
- Dilate the image to strengthen contours.
- Detect contours and extract bounding boxes.
- Merge and filter bounding boxes based on size and overlap criteria.

#### Output:
Processed bounding box images are saved in the `Processed_Images` directory. For every 4 or 5 bounding boxes, a new folder is created to organize the data.

### 2. Splitting Data into Train, Validation, and Test Sets

The bounding box images are split into:
- **70% Training**
- **15% Validation**
- **15% Testing**

#### Output:
The split datasets are saved in separate directories: `Processed_Images_Train`, `Processed_Images_Val`, and `Processed_Images_Test`.

### 3. Model Training

A Convolutional Neural Network (CNN) is trained on the processed images.

#### CNN Architecture:
- **Input Layer**: 224x224 RGB images
- **Conv2D Layers**: Extract features using convolutional filters
- **MaxPooling2D**: Reduce spatial dimensions
- **Dense Layers**: Fully connected layers with ReLU activation
- **Dropout Layers**: Prevent overfitting
- **Output Layer**: Softmax activation for multi-class classification

#### Key Features:
- Data augmentation applied to training images.
- Early stopping to prevent overfitting.
- Learning rate scheduler for optimization.

#### Output:
The trained model is saved as `signature_classification_model.h5`.

### 4. Evaluation

The model is evaluated on the test dataset to compute:
- Accuracy
- Precision, Recall, and F1-Score
- Confusion Matrix

### 5. Visualization

- Training and validation accuracy/loss curves.
- Confusion matrix heatmap.

### 6. Individual Prediction

The saved model is used to predict the class of a new image. It preprocesses the image, runs inference, and outputs the predicted label with a visualization.

### 7. Feature Extraction with HOG and SIFT (Optional)

Feature vectors are extracted using:
- **HOG (Histogram of Oriented Gradients)**
- **SIFT (Scale-Invariant Feature Transform)**

These features are padded/truncated to a fixed length (5000) and can be used for additional experiments.

---

## Installation and Usage

### Requirements:
- Python 3.7+
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Scikit-image

### Steps to Run:
1. Place raw images in the `Data` directory.
2. Run the preprocessing script:
   ```bash
   python main_script.py
   ```
3. The pipeline will process the images, train the CNN model, and save results in the respective directories.
4. Evaluate the model on test data or predict a single image using the saved model.

---

## Outputs

- **Processed Images**: Cropped bounding box images organized into folders.
- **Trained Model**: Saved in `signature_classification_model.h5`.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.
- **Training Logs**: Plots for accuracy and loss trends.

---

## Notes

- The bounding box filtering criteria (e.g., area thresholds) can be adjusted in the `process_image` function.
- The CNN architecture and hyperparameters (e.g., learning rate, batch size) can be modified in the model training section.
- For smaller datasets, consider increasing data augmentation to improve generalization.


