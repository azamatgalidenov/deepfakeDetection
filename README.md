# Deepfake Detection Project

## Overview
This project focuses on the detection of deepfake videos using Deep Learning. Deepfakes are sophisticated AI-generated videos that superimpose one person's likeness over another, creating convincing video content that appears real. Our goal is to develop a system that can accurately distinguish between genuine and artificially manipulated videos to prevent the spread of misinformation.
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/078e7a80-1a9f-4e51-a038-b1c778b5583e)

## Team Members
- Azamat Galidenov
- Temirlan Yeslamov
- Zholdas Aldanbergen
- Museong Park

## Technologies Used
- **Python**: Programming language for implementing algorithms and managing data.
- **Keras + TensorFlow**: Used for building and training the deep learning model.
- **Scikit-learn**: Utilized for additional machine learning tools for data processing and model evaluation.
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/10dc3670-0f2e-4721-8aa8-f5e48defca04)

## Dataset
We used a well-curated dataset from the Kaggle Deepfake Detection Challenge, which includes thousands of labeled images categorized as real or fake. Here is the data distribution:
- Training: Real: 5,773, Fake: 7,062
- Validation: Real: 1,364, Fake: 1,367
- Test: Real: 1,178, Fake: 1,178
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/97acbd73-6ab5-44a6-9809-c258b950466c)


## Preprocessing Steps
1. Frame Extraction: Extract frames from videos at a rate of one frame per second.
2. Face Detection: Detect and crop faces from the frames.
3. Data Arrangement: Split data into training, validation, and testing sets.
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/22583720-1b81-487f-a6ab-c4f04f71f895)


## Data Augmentation
To mitigate the imbalance in our dataset, we augmented the minority class by applying random transformations like horizontal flipping, rotation, and color enhancement.
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/18e57466-1164-46f1-be2e-c5fef7116b9e)


## Model Architecture
We utilized the VGG16 architecture, a deep convolutional network known for its high accuracy in image classification tasks. It consists of 13 convolutional layers, 5 pooling layers, and 3 dense layers.
![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/b80d7d8c-8c45-416e-9223-ad4177a666cd)


## Training
The model was trained using a split of 80% training, 10% testing and 10% validation data. The performance of the model improved significantly after tuning hyperparameters.

## Results
The final model achieved an accuracy of over 92% on the validation set. The confusion matrix from the model testing shows high precision and recall rates.

![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/a674adf6-3633-4e4c-a281-78f1016b0308)

![image](https://github.com/kazanova777/deepfakedetection/assets/117648953/383b1a96-b379-4b14-ac58-9af1a9b95121)


## Challenges and Future Work
The major challenge was handling the subtle features that distinguish real videos from deepfakes, such as inconsistencies in lighting and textures. Future improvements might include implementing more complex models and increasing the dataset size for better generalization.

## Conclusion
This project demonstrates the potential of using deep learning for detecting deepfake videos effectively, contributing to the ongoing efforts in digital media authenticity verification.
