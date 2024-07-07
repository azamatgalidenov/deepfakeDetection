import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

test_dir = 'split_dataset/test'
graph_dir = 'graph'
model_path = 'checkpoints/model_epoch_05_val_accuracy_0.96.keras'

# Load the trained model
model = load_model(model_path)

# Preprocess the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Change this to match the model's input size
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Make predictions
y_pred_prob = model.predict(test_generator)
y_pred = np.where(y_pred_prob > 0.5, 1, 0).flatten()
y_true = test_generator.classes

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = list(test_generator.class_indices.keys())

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names, graph_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
    plt.close()

# Ensure the graph directory exists
os.makedirs(graph_dir, exist_ok=True)

# Plot and save the confusion matrix
plot_confusion_matrix(cm, class_names, graph_dir)

print("Confusion matrix saved to", os.path.join(graph_dir, 'confusion_matrix.png'))