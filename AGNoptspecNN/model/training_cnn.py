import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

class cnn_trainer:
    def __init__(self):
        self.input_dim=None
        self.model_save_path = None

    def model_compile(self):
        """
        Compile the convolutional model here.
        """
        inputs = keras.layers.Input(shape=self.input_dim)
        
        #### CONVOLUTION
        x = keras.layers.Conv1D(4, 20, activation='relu')(inputs)
        x = keras.layers.MaxPooling1D(4)(x)
        x = keras.layers.Conv1D(2, 4, activation='relu')(x)
        x = keras.layers.MaxPooling1D(4)(x)
        
        #### Flatten layers
        x = keras.layers.Flatten()(x)

        #### Deep layers
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dense(16, activation='relu')(x)
        outputs = keras.layers.Dense(3, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['sparse_categorical_accuracy',])
    
    def train(self, verbose):
        """
        Train the model here. The training data should be passed as arguments.
        returns: "accuracy", "loss"
        """
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        self.history  = self.model.fit(self.X_train, self.y_train, 
                                       epochs=self.epochs, 
                                       shuffle=True, validation_split=0.2,
                                       verbose=verbose,
                                       callbacks=[early_stop] 
                                       )
        return self.history 

    def performance(self):
        """
        Plot the performance of the model.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(self.history.history['sparse_categorical_accuracy'], label='training: accuracy')
        axs[0].plot(self.history.history['val_sparse_categorical_accuracy'], label='validation: accuracy')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Sparse Categorical Accuracy')
        axs[0].legend()

        axs[1].plot(self.history.history['loss'], label='training: loss')
        axs[1].plot(self.history.history['val_loss'], label='validation: loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()


    def tests(self, which=None):
        """
        Test the model here. The test data should be passed as arguments.
        """
        if which in [None,"test"]:
            y_pred = self.model.predict(self.X_test).T
            self.y_pred_labels = [np.argmax(i) for i in y_pred.T]
            self.cm = tf.math.confusion_matrix(labels=self.y_test, predictions=self.y_pred_labels)
            print("===== Metrics of the test-data =====")
            print(classification_report(self.y_test, self.y_pred_labels))
        elif which == "train":
            y_pred = self.model.predict(self.X_train).T
            self.y_pred_labels = [np.argmax(i) for i in y_pred.T]
            self.cm = tf.math.confusion_matrix(labels=self.y_train, predictions=self.y_pred_labels)
            print("===== Metrics of the train-data =====")
            print(classification_report(self.y_train, self.y_pred_labels))
        else:
            print("Invalid dataset.")

    def model_save(self, path, filename):                 ## Look out for this
        """
        Save the trained model to the specified path.
        """
        try:
            self.model.save(f"{path}{filename}")
        except FileNotFoundError:
            print(f"Making the directory")
            os.mkdir(path)
            self.model.save(f"{path}{filename}")
