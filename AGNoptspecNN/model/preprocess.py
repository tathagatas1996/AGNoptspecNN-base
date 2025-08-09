import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

class preprocess_1D_spectra:
    def __init__(self, path):
        self.nx = 2000
        self.path = path
        self.files = os.listdir(self.path) ## Look out for this ## THESE CAN BE ZENODO PATHS.
    
    def spectral_type_classify(self, type):
        """
        AGN type
        """
        if type == 'Type-1':
            y_val = 0.

        elif type == 'Type-1.9':
            y_val = 1.

        elif type == 'Type-2':
            y_val = 2.

        return y_val
    
    def read_files(self):
        self.X = []
        self.y = []
        
        for i in range(len(self.files)):
            DATASET = fits.open(f"{self.path}{self.files[i]}") ## Look out for this
            type =  DATASET[1].header['LABEL']
            
            y_val = self.spectral_type_classify(type)

            wave = DATASET[1].data['wavelength']
            flux = DATASET[1].data['flux']

            flux_func1 = interp1d(wave, flux)
            wave_st = np.linspace(min(wave), max(wave), self.nx)

            flux_std = flux_func1(wave_st)

            flux_std = flux_std/np.mean(flux_std)

            self.X.append(flux_std)
            self.y.append(y_val)

        
        self.X = np.array(self.X).reshape( len(self.files), self.nx, 1)
        self.y = np.array(self.y)

        return (self.X, self.y)
