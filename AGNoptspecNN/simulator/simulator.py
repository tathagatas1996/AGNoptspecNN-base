import os
import numpy as np
import random as random
from scipy.interpolate import interp1d
import scipy.constants as cns
from .iron_pseudo_continuum import *
from .lines import emission_lines, line_list
from importlib.resources import files


#######################################################

def gauss(x, normalization, center, sigma):
    """
    Parameters
    ----------
    x : array-like
        usually Wavelengths in Angstroms.
    normalization : float
        Normalization factor for the Gaussian function.
    center : float
        Center of the Gaussian function, typically the central wavelength of the emission line.
    sigma : float
        Standard deviation of the Gaussian function, related to the line width.
    """
    ff = normalization*np.exp(-0.5*((x - center)/sigma)**2)
    return ff 


class gaussian_noise:
    def __init__(self, snr, amplitude):
        """
        Parameters
        ----------
        snr : float
            Signal-to-noise ratio for the noise generation.
        amplitude : float
            Amplitude of the signal, used to determine the noise level.
            The noise level is calculated as amplitude divided by the SNR.
        """
        self.snr = snr
        self.amplitude = amplitude 
        self.noise_level = self.amplitude/self.snr
        
    def noise_spectrum(self, x):
        """
        Gaussian Noise
        """
        ff = np.random.normal( 0, self.noise_level, size=len(x))
        return ff


class quasar_continuum:
    def __init__(self):
        self.lamda_0 = 5100.0        
    def powerlaw(self, x, normalization, index):
        """
        Parameters
        ----------
        x : array-like
            Wavelengths in Angstroms.
        normalization : float
            Normalization factor for the power law.
        index : float
            Power law index, typically between 0.5 and 3.0 for quasar continua.
            Note that the "-" sign is explicit. This means for +ve value of index you are getting a monotonically decreasing function.
        """
        ff = normalization*(x/self.lamda_0)**-index
        return ff


class galaxy_morph:
    """This class is used to generate the galaxy template spectrum based on the morphological type of the galaxy."""
    def __init__(self, type):
        """
        Parameters
        ----------
        type : str
            The morphological type of the galaxy. It can be one of the following:
            - "E0" or "e0" for Elliptical galaxies
            - "S0" or "s0" for Lenticular galaxies
            - "Sa" or "sa" for Spiral galaxies of type Sa
            - "Sb" or "sb" for Spiral galaxies of type Sb
            - "Sc" or "sc" for Spiral galaxies of type Sc

        The templates are taken from Mannucci et al. 2001
        """
        self.type = type

        path_gal = files("AGNoptspecNN.data.templates.galaxy").joinpath("ggaltempl_mannucci01.txt") 
        hg_in = np.loadtxt(str(path_gal)).T

        in_wavelength = hg_in[0]*1e4 ## in 
        flux_E0 = hg_in[1]
        flux_s0 = hg_in[2]
        flux_sa = hg_in[3]
        flux_sb = hg_in[4]
        flux_sc = hg_in[5]

        if type == "E0"or type=="e0":
            self.gal_template = interp1d( in_wavelength, flux_E0, fill_value='extrapolate')
        elif type == "S0" or type=="s0": 
            self.gal_template = interp1d( in_wavelength, flux_s0,fill_value='extrapolate')
        elif type == "Sa" or type=="sa":
            self.gal_template = interp1d( in_wavelength, flux_sa,fill_value='extrapolate')
        elif type == "Sb" or type=="sb":
            self.gal_template = interp1d( in_wavelength, flux_sb,fill_value='extrapolate')
        elif type == "Sc" or type=="sc":
            self.gal_template = interp1d( in_wavelength, flux_sc,fill_value='extrapolate')
        else:
            print("invalid input")
        
    def gal_spec(self, x, cgal):
        return cgal*self.gal_template(x)


class Emission_lines:
    def __init__(self, type):
        self.type = type
        
        self.vbroad_min = 1000.
        self.vbroad_max = 5000.

        self.narrow_min = 200.
        self.narrow_max = 400.

        self.sigma_narrow_velocity_by_c = np.random.randint(self.narrow_min, self.narrow_max)*1000/cns.c
        self.sigma_broad_velocity_by_c  = np.random.randint(self.vbroad_min, self.vbroad_max)*1000/cns.c

        
    def process_the_emission_line_parameters(self):
        self.waveN , self.waveB  =  [],[]
        self.sigmaN, self.sigmaB =  [],[]
        self.amplN , self.amplB  =  [],[]
        
        for i in range( len(emission_lines) ):
            
            if self.type == "Type-1":
                if emission_lines[i+1][2] == "n" or emission_lines[i+1][2] == "bn":
                    self.waveN.append(  emission_lines[i+1][1] ) 
                    self.sigmaN.append( emission_lines[i+1][1]*
                                        self.sigma_narrow_velocity_by_c )
                    self.amplN.append(  emission_lines[i+1][3] * (1+0.5*np.random.random()) ) # introduce some randomness in the amplitude
                if emission_lines[i+1][2] == "b" or emission_lines[i+1][2] == "bn":
                    self.waveB.append(  emission_lines[i+1][1] ) 
                    self.sigmaB.append( emission_lines[i+1][1]*
                                        self.sigma_broad_velocity_by_c )
                    self.amplB.append(  emission_lines[i+1][3] * (1+0.5*np.random.random()) ) # introduce some randomness in the amplitude

            
            elif self.type == "Type-1.9":
                if emission_lines[i+1][2] == "n" or emission_lines[i+1][2] == "bn":
                    self.waveN.append(  emission_lines[i+1][1] ) 
                    self.sigmaN.append( emission_lines[i+1][1]*
                                        self.sigma_narrow_velocity_by_c )
                    self.amplN.append(  emission_lines[i+1][3] * (1+0.5*np.random.random()) ) # introduce some randomness in the amplitude   
                if emission_lines[i+1][2] == "bn" and emission_lines[i+1][0]=="Halpha":
                    self.waveB.append(  emission_lines[i+1][1] ) 
                    self.sigmaB.append( emission_lines[i+1][1]*
                                        self.sigma_broad_velocity_by_c )
                    self.amplB.append(  emission_lines[i+1][3] * (1+0.5*np.random.random()) ) # introduce some randomness in the amplitude

            
            elif self.type == "Type-2":
                if emission_lines[i+1][2] == "n" or emission_lines[i+1][2] == "bn":
                    self.waveN.append(  emission_lines[i+1][1] ) 
                    self.sigmaN.append( emission_lines[i+1][1]*
                                        self.sigma_narrow_velocity_by_c )
                    self.amplN.append(  emission_lines[i+1][3]*(1+0.5*np.random.random()) ) # introduce some randomness in the amplitude


    def line_complex(self, x, norm): 
        self.process_the_emission_line_parameters()
        
        if self.type == "Type-1":
            SSn,SSb = 0,0
            for i in range(len(self.waveN)):
                SSn += gauss(x, self.amplN[i], self.waveN[i], self.sigmaN[i])
            for i in range(len(self.waveB)):
                SSb += gauss(x, self.amplB[i], self.waveB[i], self.sigmaB[i])
            return norm*(SSn + SSb)
            
        elif self.type == "Type-1.9":
            SSn,SSb = 0,0
            for i in range(len(self.waveN)):
                SSn += gauss(x, self.amplN[i], self.waveN[i], self.sigmaN[i])
            for i in range(len(self.waveB)):
                SSb += gauss(x, self.amplB[i], self.waveB[i], self.sigmaB[i])
            return norm*(SSn + SSb)
            
        elif self.type == "Type-2":
            SS = 0
            for i in range(len(self.waveN)):
                SS += gauss(x, self.amplN[i], self.waveN[i], self.sigmaN[i])
            return norm*SS    


def total_spectrum_rest(x, type):
    """
    Simulates the total rest-wavelength spectrum for a given type of AGN or galaxy.

    Parameters
    ----------
    x : array-like
        Wavelengths in Angstroms.
    type : str
        The type of the AGN spectrum to simulate. It can be one of the following:
        - "Type-1" for Type-1 AGN
        - "Type-1.9" for Type-1.9 AGN
        - "Type-2" for Type-2 AGN
        - Any other string for a galaxy template spectrum without AGN features.
    """
    ################### Galaxy template ###################
    galaxy_types = ["e0", "s0", "sa", "sb", "sc"]
    gal_type = random.choice(galaxy_types)
    galaxy = galaxy_morph(gal_type)
    cgal = np.random.uniform(low=0.2, high=1.5)
    
    if type == "Type-1":
        ################### Lines ########################
        lines = Emission_lines(type)
        norm_l = np.random.uniform(low=0.5, high=1.0)
        
        ################## Quasar continuum ###################        
        quasar_cont = quasar_continuum()
        amp = np.random.uniform(low=0.4, high=2.0)
        p   = np.random.uniform(low=0.5, high=4.0)
        
        ################# Iron pseudo-continuum ######################
        v = random.choice( fe_velocities )
        iron_template = iron_template_optical(v=v)
        cfe_norm = np.random.uniform(low=0.0, high=0.02)

        ######################################################
        ff = galaxy.gal_spec(x, cgal) + quasar_cont.powerlaw(x, amp, p) + iron_template.feII_temp_option1(x, cfe_norm) + lines.line_complex(x, norm_l)

        ################# Noise spectrum #####################
        snr = np.random.uniform(low=10, high=80.0)
        noise = gaussian_noise(snr, np.std(ff))
        ######################################################
        return ff + noise.noise_spectrum(x)
        
    elif type == "Type-2" or type=="Type-1.9":
        ################### Lines ##############################
        lines = Emission_lines(type)
        norm_l = np.random.uniform(low=0.5, high=1.0)
        
        ################## Quasar continuum ###################        
        quasar_cont = quasar_continuum()
        amp = np.random.uniform(low=0, high=0.2)*cgal
        p = np.random.uniform(low=0.5, high=3.0)
        
        ######################################################
        ff = galaxy.gal_spec(x, cgal) + quasar_cont.powerlaw(x, amp, p) + lines.line_complex(x, norm_l)

        ################ Noise spectrum #####################
        snr = np.random.uniform(low=20, high=80.0)
        noise = gaussian_noise(snr, np.std(ff))
        
        #####################################################
        return ff + noise.noise_spectrum(x)
        
    else:
        ff =  galaxy.gal_spec(x, cgal)
    return ff


def total_spectra_redshifted(type, z, lamda_min, del_lamda, Npix):
    """
    This functions returns the final training/test data:
    ------------------------------
    type: Type-1, Type-1.9 or Type-2
    z: redshift
    lamda_min: minimum wavelength
    del_lamda: interval
    Npix: number of pixels
    ------------------------------
    Returns an 1-D array of flux
    """
    lamda_max = lamda_min + Npix * del_lamda
    lamda = np.arange(lamda_min, lamda_max, del_lamda)    
    flux = total_spectrum_rest(lamda/(1+z), type)
    return (lamda, flux)