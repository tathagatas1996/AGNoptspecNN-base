import numpy as np
import random as random
from astropy.io import fits
from .simulator import *
from AGNoptspecNN.config import *

header_keywords = ["CDELT0", "CDELT1", "NPIX", "Redshift", "Spectral_Type"]

Npix = [500, 1000, 2000, 4000, 5000, 8000] # number of wavelength bins

def get_dlam(npix):
    if npix == 500:
        dlam = random.uniform(9.0, 10.0) # 500
    elif npix == 1000:
        dlam = random.uniform(3.5, 4.8) # 1000
    elif npix == 2000:
        dlam = random.uniform(2.5, 3.0) # 2000
    elif npix == 4000:
        dlam = random.uniform(1.5, 2.5) # 4000
    elif npix == 5000:
        dlam = random.uniform(1.0, 2.0) # 5000
    elif npix == 8000:
        dlam = random.uniform(0.6, 0.9) # 8000
    return dlam


def simulate_spectrum_optical(type, status, path, filename):
    """
    Parameters
    ----------
    type: str, optical type of AGN
    status: str, Should the file be written, if status = "w" initiates the process for writing.
    path: directory
    filename: filename
    """
    if status == "w":
        if path is None or filename is None:
            raise ValueError("When status='w', both 'path' and 'filename' must be provided.")

    npix = random.choice( Npix )
    lamda_min = random.randint(3500, 4500)
    dlam = get_dlam(npix)

    z = random.uniform(0.0001, 0.2) ## redshift for mostly nearby Seyferts

    wavelength, flux = total_spectra_redshifted(type, z, lamda_min, dlam, npix)

    if status == 'w':
        col1 = fits.Column(name='wavelength', format='D', array=wavelength)
        col2 = fits.Column(name='flux', format='D', array=flux)
        hdu = fits.BinTableHDU.from_columns([col1, col2])

        hdu.header['LABEL']    = (type, 'Spectral type')
        hdu.header['CDELT0']   = (lamda_min, 'Simulation/instrument type')
        hdu.header['CDELT1']   = (dlam, 'Simulation/instrument type')
        hdu.header['redshift'] = (z, 'redshift')
        hdu.header['NPIX'] = (npix, 'number of wavelength intervals')

        hdu.writeto(f"{path}{filename}", overwrite=True)

    return (wavelength, flux)