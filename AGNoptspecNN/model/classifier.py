from importlib.resources import files
import random
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow import keras
from AGNoptspecNN.model.preprocess import *


class ClassificationModel:
    def __init__(self):
        """
        ==========
        Using three models here:   Dimensions:      (None, nx ,  1) 
        model_v1_ninterp500  : The model which uses (None, 500,  1) as input. 
        model_v1_ninterp1000 : The model which uses (None, 1000, 1) as input.
        model_v1_ninterp2000 : The model which used (None, 2000, 1) as input.
        ==========
        """
        model_path1 = files("AGNoptspecNN.saved_model").joinpath("model_v1_ninterp500.keras")
        model_path2 = files("AGNoptspecNN.saved_model").joinpath("model_v1_ninterp1000.keras")
        model_path3 = files("AGNoptspecNN.saved_model").joinpath("model_v1_ninterp2000.keras")

        self.model1 = tf.keras.models.load_model(str(model_path1))
        self.model2 = tf.keras.models.load_model(str(model_path2))
        self.model3 = tf.keras.models.load_model(str(model_path3))

        self.nx1 = 500  # 
        self.nx2 = 1000 #
        self.nx3 = 2000 #

    def reshape_spec(self, wave, X, N):
        """
        ============
        wave : Wavelength 
        X    : Flux
        N    : Number of interpolated spectral points
        ============
        returns: reshaped spectra. 
        """
        flux_func1 = interp1d(wave, X)
        wave_st  = np.linspace(min(wave), max(wave), N)
        flux_std = flux_func1(wave_st)
        flux_std = flux_std/np.mean(flux_std)
        return (wave_st, flux_std)

    def vote_array(self, arr):
        unique, counts = np.unique(arr, return_counts=True)
        if np.any(counts >= len(arr)/2):     
            return unique[np.argmax(counts)]
        else:
            return np.random.choice(arr) 

    def classify_the_spectrum(self, Spectrum):
        """
        Classify a single spectrum here. The spectrum should be passed as an argument.
        """
        wave = Spectrum[:,0]
        X = Spectrum[:,1]

        wave_1, X_1 = self.reshape_spec(wave, X, 500)
        wave_2, X_2 = self.reshape_spec(wave, X, 1000)
        wave_3, X_3 = self.reshape_spec(wave, X, 2000)

        X_for_prediction1 = X_1.reshape(1,-1,1)
        X_for_prediction2 = X_2.reshape(1,-1,1)
        X_for_prediction3 = X_3.reshape(1,-1,1)

        ypred1 = np.argmax((self.model1.predict(X_for_prediction1, verbose=0).T))
        ypred2 = np.argmax((self.model2.predict(X_for_prediction2, verbose=0).T))
        ypred3 = np.argmax((self.model3.predict(X_for_prediction3, verbose=0).T))

        prediction_model  = np.array([1,2,3])
        model_label       = np.array(["Model-1", "Model-2", "Model-3"])  
        y_prediction_list = np.array([ypred1, ypred2, ypred3]) # Prediction from individual models
        
        ypred = self.vote_array(y_prediction_list) # Final prediction

        print("===== Message from classifier =====")
        if ypred == 0:
            print(f"Network output {ypred}: thus Type 1 AGN.")
        elif ypred == 1:
            print(f"Network output {ypred}: thus Type 1.9 AGN.")
        elif ypred == 2:
            print(f"Network output {ypred}: thus Type 2 AGN.")
        print("===================================")

        return (ypred, prediction_model, model_label, y_prediction_list)

#=============================================================================#
def testing_files(nfiles, SPECTRA_DIR):
    """
    This function is for the training-testing data shipped with the model.
    """
    Np=[500, 1000, 2000] 

    p1 = preprocess_1D_spectra( f"{SPECTRA_DIR}/type1/"   ) # These can be zenodo paths. Change accordingly.
    p2 = preprocess_1D_spectra( f"{SPECTRA_DIR}/type1pt9/") # These can be zenodo paths. Change accordingly.
    p3 = preprocess_1D_spectra( f"{SPECTRA_DIR}/type2/"   ) # These can be zenodo paths. Change accordingly.

    each = int((1/3)*nfiles)

    p1.files = random.sample(p1.files, each) 
    p2.files = random.sample(p2.files, each) 
    p3.files = random.sample(p3.files, each)

    p1.nx = Np[0] 
    p2.nx = Np[0]
    p3.nx = Np[0]

    X11, y11 = p1.read_files()
    X21, y21 = p2.read_files()
    X31, y31 = p3.read_files()

    p1.nx = Np[1] 
    p2.nx = Np[1]
    p3.nx = Np[1]

    X12, y12 = p1.read_files()
    X22, y22 = p2.read_files()
    X32, y32 = p3.read_files()

    p1.nx = Np[2] 
    p2.nx = Np[2]
    p3.nx = Np[2]

    X13, y13 = p1.read_files()
    X23, y23 = p2.read_files()
    X33, y33 = p3.read_files()

    X1 = np.concatenate( (X11, X21, X31), axis=0)
    y1 = np.concatenate( (y11, y21, y31) )

    X2 = np.concatenate( (X12, X22, X32), axis=0)
    y2 = np.concatenate( (y12, y22, y32) )

    X3 = np.concatenate( (X13, X23, X33), axis=0)
    y3 = np.concatenate( (y13, y23, y33) )

    if np.array_equal(y1,y2) and np.array_equal(y2,y3):
        return (X1, X2, X3, y1)
    else:
        print("Mismatch observed in the Labels !!")
        return None 


def tests(X_group1, X_group2, X_group3, y):
    """
    X_group1 : 500  
    X_group2 : 1000
    X_group3 : 2000
        The same datasets wavelength array sampled to have 500, 1000, and 2000 points 
    y: 
        real label 
    ---------------------------
    returns: (y= real label, ypred = predicted label from the three models)
    """
    model_define = ClassificationModel()

    y_pred1 = model_define.model1.predict(X_group1).T
    y_pred2 = model_define.model2.predict(X_group2).T
    y_pred3 = model_define.model3.predict(X_group3).T

    y_pred_labels1 = [np.argmax(i) for i in y_pred1.T]
    y_pred_labels2 = [np.argmax(i) for i in y_pred2.T]
    y_pred_labels3 = [np.argmax(i) for i in y_pred3.T]

    ypred = np.zeros(len(y_pred_labels1))
    
    for i in range(ypred.size):
        ypred[i] = model_define.vote_array(np.array([y_pred_labels1[i], y_pred_labels2[i], y_pred_labels3[i]]))
    
    return (y, ypred)