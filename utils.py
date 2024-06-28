import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import cv2
import os
import json
import tensorflow as tf
import numpy as np


class Plotter:
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, title=None, normalize=None, path=None):
        """
        Plots confusion matrix.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param classes: feature names labels.
        :param title: Title of the plot.
        :param normalize: Normalize the confusion matrix.
        :param path: Path to save the plot.
        :return: None
        """
        if normalize==None:
            values_format = 'd'
        else:
            values_format = '.3f'

        ConfusionMatrixDisplay.from_predictions(
            y_pred=y_pred,
            y_true=y_true,
            cmap='Greys',
            normalize=normalize,
            display_labels=classes,
            values_format=values_format
            )
        
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')   
        plt.savefig(path)


    def plot_training_loss_and_acc(self, hist, path, title=None):
        """
        Plots training loss and accuracy.
        :param hist: History object from model.fit.
        :param title: Title of the plot.
        :param path: Path to save the plot.
        :return: None
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        fig.tight_layout(pad=5.0)
        fig.suptitle(title)
        ax1.set_title('Loss')
        ax1.set_ylabel('Loss (training and validation)')
        ax1.set_xlabel('Epoch')
        ax1.plot(hist.history['loss'], 'r', label='Loss')
        ax1.plot(hist.history['val_loss'], 'b', label='Validation Loss')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2.set_title('Accuracy')
        ax2.set_ylabel('Accuracy (training and validation)')
        ax2.set_xlabel('Epoch')
        ax2.plot(hist.history['accuracy'], 'r', label='Accuracy')
        ax2.plot(hist.history['val_accuracy'], 'b', label='Validation Accuracy')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        # save fig
        plt.savefig(path)


class Preprocessor:
    def read_json(self, json_file_path):
        """
        Loading the json file
        """

        with open(json_file_path, "r") as f:
            contents = json.load(f)

        return contents
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0  # Normalize to 0-1
        return img
    
    def get_data(self, path):
        # Preprocess train images
        metadata_path = '/home/mp30265/deepfake_detection/data/train_sample_videos/metadata.json'
        metadata = self.read_json(metadata_path)

        preprocessed_train_images = []
        train_labels = []
        for video_folder in os.listdir(os.path.join(path, 'train')):
            augmented_folder_path = os.path.join(path, 'train', video_folder)
            label = 1 if metadata[video_folder + '.mp4']['label'] == 'FAKE' else 0
            for img_file in os.listdir(augmented_folder_path):
                img_path = os.path.join(augmented_folder_path, img_file)
                preprocessed_train_images.append(self.preprocess_image(img_path))
                train_labels.append(label)

        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
        
        """
        # Preprocess test images
        preprocessed_test_images = []
        for video_folder in os.listdir(os.path.join(path, 'test')):
            augmented_folder_path = os.path.join(path, 'test', video_folder)
            for img_file in os.listdir(augmented_folder_path):
                img_path = os.path.join(augmented_folder_path, img_file)
                preprocessed_test_images.append(preprocess_image(img_path))
        """

        train_images = np.array(preprocessed_train_images)
        train_labels = np.array(train_labels)


        x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, shuffle=True, random_state=42)
        return x_train, x_val, y_train, y_val
    