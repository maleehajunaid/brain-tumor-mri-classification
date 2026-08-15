# Brain Tumor MRI Classification

A deep learning project for classifying brain MRI images into four categories using transfer learning with an ImageNet-pretrained MobileNetV2 model.

## Classes

- Glioma
- Meningioma
- Pituitary
- No Tumor

## Project Overview

This project uses transfer learning for multiclass brain tumor classification from MRI images. A pretrained MobileNetV2 model is used as a feature extractor, while a custom classification head is trained on the brain MRI dataset.

Grad-CAM is also used to visualize the regions of MRI images that contribute to the model's predictions.

## Methodology

1. Load the Brain Tumor MRI Dataset.
2. Preprocess and resize MRI images to 224 × 224 pixels.
3. Normalize image pixel values.
4. Apply data augmentation to training images.
5. Split the training data into training and validation sets.
6. Use ImageNet-pretrained MobileNetV2 as the backbone.
7. Freeze the pretrained MobileNetV2 layers.
8. Add a custom classification head for four classes.
9. Train the classification head for 10 epochs.
10. Evaluate the trained model on a separate testing dataset.
11. Apply Grad-CAM for visual model interpretation.
12. Save the trained model for later use.

## Model Architecture

**Input MRI Image**  
↓  
**MobileNetV2 (ImageNet pretrained)**  
↓  
**Frozen Feature Extractor**  
↓  
**Global Average Pooling**  
↓  
**Dense Layer (128 units)**  
↓  
**Softmax Output Layer**  
↓  
**4 MRI Classes**

## Dataset

The project uses the Brain Tumor MRI Dataset obtained from Kaggle.

The dataset contains four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The training directory is divided into 80% training data and 20% validation data. A separate testing directory is used for final evaluation.

## Explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to provide visual explanations for the model's predictions by highlighting important regions of an MRI image that contribute to the predicted class.

## Technologies Used

Python, TensorFlow, Keras, MobileNetV2, NumPy, Matplotlib, and Google Colab.

## Project Workflow

Brain MRI Dataset → Preprocessing → Data Augmentation → Training/Validation Split → Pretrained MobileNetV2 → Frozen Backbone → Custom Classification Head → Training → Testing & Evaluation → Grad-CAM Visualization

## Results

The complete training process, evaluation results, and Grad-CAM visualizations are available in the Colab Notebook included in this repository.






