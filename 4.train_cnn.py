import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Paths to the datasets
train_dir = 'split_dataset/train'
val_dir = 'split_dataset/val'
test_dir = 'split_dataset/test'
graph_dir = 'graph'
checkpoint_dir = 'checkpoints'

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

# Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 5
learning_rate = 1e-4  # Learning rate tuned from 2e-4 to 1e-4 result, result - val_accuracy from 0.84 to 0.95

# Create data generators without augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create train, validation, and test generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Print class indices to ensure correct mapping
class_indices = train_generator.class_indices
print(class_indices)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Build the model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze some of the base model layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.2f}.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model with the callbacks
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save model history plots
def save_plot(history, metric, filename):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(graph_dir, filename))
    plt.close()

save_plot(history, 'accuracy', 'model_accuracy.png')
save_plot(history, 'loss', 'model_loss.png')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)

# Predict using the model
Y_pred = model.predict(test_generator)
y_pred = np.round(Y_pred).astype(int).flatten()
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
plt.close()

# Train data distribution
labels, counts = np.unique(train_generator.classes, return_counts=True)
class_labels = [key for key, value in train_generator.class_indices.items()]
sns.barplot(x=class_labels, y=counts)
plt.title('Train Data Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig(os.path.join(graph_dir, 'train_data_distribution.png'))
plt.close()

print(f'Test Accuracy: {test_acc:.4f}')
