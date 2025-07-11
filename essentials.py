import os
import numpy as np
import scipy.constants as cns
from line_data import *
import random as random
from scipy.interpolate import interp1d

#######################################################

def gauss(x, normalization, center, sigma):
    ff = normalization*np.exp(-0.5*((x - center)/sigma)**2)
    return ff 


class gaussian_noise:
    def __init__(self, snr, amplitude):
        self.snr = snr
        self.amplitude = amplitude 
        self.noise_level = self.amplitude/self.snr
        
    def noise_spectrum(self, x):
        ff = np.random.normal( 0, self.noise_level, size=len(x))
        return ff


class quasar_continuum:
    def __init__(self):
        self.lamda_0 = 5100.0        
    def powerlaw(self, x, normalization, index):
        ff = normalization*(x/self.lamda_0)**-index
        return ff


class galaxy_morph:
    """This class is used to generate the galaxy template spectrum based on the morphological type of the galaxy."""
    def __init__(self, type):
        self.type = type
        hg_in = np.loadtxt('galaxy_template/ggaltempl_mannucci01.txt').T
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


def total_spectrum(x, type):
    ################### Galaxy template ###################
    galaxy_types = ["e0", "s0", "sa", "sb", "sc"]
    gal_type = random.choice(galaxy_types)
    galaxy = galaxy_morph(gal_type)
    cgal = np.random.uniform(low=0.2, high=2.0)
    
    if type == "Type-1":
        ################### Lines ########################
        lines = Emission_lines(type)
        norm_l = np.random.uniform(low=0.5, high=1.0)
        
        ################## Quasar continuum ###################        
        quasar_cont = quasar_continuum()
        amp = np.random.uniform(low=0.4, high=2.0)
        p   = np.random.uniform(low=0.5, high=3.0)
        
        ######################################################
        ff = galaxy.gal_spec(x, cgal) + quasar_cont.powerlaw(x, amp, p) + lines.line_complex(x, norm_l)

        ################# Noise spectrum #####################
        snr = np.random.uniform(low=20, high=80.0)
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