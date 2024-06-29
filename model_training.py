from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np

from utils import Preprocessor, Plotter

def create_model():
	
    CLASSES = ["REAL", "FAKE"]

    baseModel = VGG16(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(CLASSES), activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    #model.summary()
    return model



def train_model(path):
    model = create_model()

    preprocessor = Preprocessor()
    x_train, x_val, y_train, y_val = preprocessor.get_data(path)

    #early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
    hist = model.fit(x_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping], validation_data=(x_val, y_val))

    save_model(model, 'deepfake_detector')
    return hist


def save_model(model, model_name):
    model_path = '/home/mp30265/deepfake_detection/model/' + model_name + '.keras'
    tf.keras.models.save_model(model, model_path)

def save_confusion_matrix(path):
    processor = Preprocessor()
    plotter = Plotter()

    x_val, y_val = processor.get_test_data(path)
    
    model = load_model('/home/mp30265/deepfake_detection/model/deepfake_detector.keras')
    pred = model.predict(x_val)
    pred = np.argmax(pred, axis=1)


    plotter.plot_confusion_matrix(y_val, pred, path="/home/mp30265/deepfake_detection/figs/confusion_matrix.png")


path = "/home/mp30265/deepfake_detection/data/augmented_faces"
#history = train_model(path)

#plot loss and acc
plotter = Plotter()
#plotter.plot_training_loss_and_acc(history, path="/home/mp30265/deepfake_detection/figs/loss_acc.png")

test_path = "/home/mp30265/deepfake_detection/data/cropped_faces"
#save_confusion_matrix(test_path)

#get dataset distribution
save_path = '/home/mp30265/deepfake_detection/figs/distribution_test_really.png'
plotter.plot_dataset_distribution(test_path, save_path)
