import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Class preprocess_1D_spectra:
"""
class preprocess_1D_spectra:
    def __init__(self, path):
        self.nx = 500 # Number of points to which the spectra will be interpolated for standardization
        self.path = path 
        self.files = os.listdir(path)

    def standardize_the_dataset(self, x, y):
        """
        Standardize the dataset by interpolating the x and y values to a fixed number of points (self.nx).
        This is done to ensure that all spectra have the same number of points for training the neural
        """
        f = interp1d(x,y)
        x = np.linspace(min(x), max(x), self.nx)
        y = f(x)
        return (x,y)

    def reshape_for_cnn(self,X):
        """ Reshape the data for convolutional neural networks."""
        X = np.array(X)
        shape_Nspec = X.shape[0]
        shape_Npix = X.shape[1]
        X = X.reshape((shape_Nspec, shape_Npix, 1))
        return X

    def preprocess_data(self, nn):
        """
        Different types of datastructures for different types of neural networks.
        ANN: 2D array of shape (Nspec, Npix)
        CNN: 3D array of shape (Nspec, Npix, 1)
        """
        self.X = []
        self.y = []
        
        for i in range(len(self.files)):
            with open(self.path + self.files[i], 'r') as file:
                data = json.load(file)
                x, Xt = self.standardize_the_dataset(data['x'], data['y'])
                self.X.append(Xt)
                self.y.append(data['t'])
        
        self.y = np.array(self.y)
        
        if nn == "ANN" or nn=="ann":
            self.X = np.array(self.X)
        
        elif nn == "CNN" or nn=="cnn":
            self.X = self.reshape_for_cnn(self.X)


"""
The convolutional neural network class. Use it to train and test the model.
Also use it to load and use a pre-trained model for classification.
"""

class convagnNN:
    def __init__(self):
        self.epochs = 10
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.input_dim = None
        

    def model_compile(self):
        """
        Compile the convolutional model here. 
        """
        self.model = keras.Sequential([
            ## Convolution layers
            keras.layers.Conv1D(4, 20, activation='relu', input_shape=(self.input_dim)),
            keras.layers.MaxPooling1D(4),
            keras.layers.Conv1D(2, 4, activation='relu'),
            keras.layers.MaxPooling1D(4),
            
            ## Dense Neural networks
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, verbose):
        """
        Train the model here. The training data should be passed as arguments.
        """
        self.history  = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, verbose=verbose)

    def performance(self):
        """
        Plot the performance of the model.
        """
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['loss'], label='loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.legend()

    def tests(self):
        """
        Test the model here. The test data should be passed as arguments.
        """
        y_pred = self.model.predict(self.X_test).T
        self.y_pred_labels = [np.argmax(i) for i in y_pred.T]
        self.cm = tf.math.confusion_matrix(labels=self.y_test, predictions=self.y_pred_labels)
        print(classification_report(self.y_test, self.y_pred_labels))


    def model_save(self, path):
        """
        Save the trained model to the specified path.
        """
        try:
            self.model.save(path+'/agn_optical.keras')
        except FileNotFoundError:
            print(f"Making the directory")
            os.mkdir(path)
            self.model.save(path+'/agn_optical.keras')
    
    def load_model(self, path):
        """
        Load the model from the specified path.
        """
        self.model = tf.keras.models.load_model(path)
    
    
    def classify_the_spectrum(self, X):
        """
        Classify a single spectrum here. The spectrum should be passed as an argument.
        """
        X_for_prediction = X.reshape(1,-1,1)
        type = np.argmax((self.model.predict(X_for_prediction, verbose=0).T))
        if type == 0:
            print("Type 1 AGN")
        elif type == 1:
            print("Type 1.9 AGN")
        elif type == 2:
            print("Type 2 AGN")
        return type