from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

from utils import Preprocessor, Plotter

def create_model():
	
    CLASSES = ["FAKE", "REAL"]

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
    hist = model.fit(x_train, y_train, batch_size=32, epochs=150, callbacks=[early_stopping], validation_data=(x_val, y_val))

    save_model(model, 'deepfake_detector')
    return hist


def save_model(model, model_name):
    model_path = '/home/mp30265/deepfake_detection/model' + model_name + '.keras'
    tf.keras.models.save_model(model, model_path)


path = "/home/mp30265/deepfake_detection/data/augmented_faces"
history = train_model(path)

#plot loss and acc
plotter = Plotter()
plotter.plot_training_loss_and_acc(history, path="/home/mp30265/deepfake_detection/figs/loss_acc.png")
