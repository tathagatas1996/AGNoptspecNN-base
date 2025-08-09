import numpy as np
from scipy.interpolate import interp1d
from AGNoptspecNN.config import iron_template_path
from importlib.resources import files

####### Available FeII velocities ########
fe_velocities = np.array([
                           700, 800, 900, 
                           1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 
                           2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800
                        ]) # km/s
############################################

class iron_template_optical:
    def __init__(self, v):
        """
        Parameters
        ----------
        path : str
            Path to the directory containing the FeII template files.
        v : int
            Velocity parameter used to select the appropriate FeII template files.
        """
        self.v = v

        str1 = files(f"AGNoptspecNN.data.templates.iron.FeII_template_4000_5500.{v}FeII").joinpath("fe_f.txt")    # Group F
        str2 = files(f"AGNoptspecNN.data.templates.iron.FeII_template_4000_5500.{v}FeII").joinpath("fe_g.txt")    # Group G
        str3 = files(f"AGNoptspecNN.data.templates.iron.FeII_template_4000_5500.{v}FeII").joinpath("fe_IZw1.txt") #IZw1
        str4 = files(f"AGNoptspecNN.data.templates.iron.FeII_template_4000_5500.{v}FeII").joinpath("fe_p.txt")    # Group P
        str5 = files(f"AGNoptspecNN.data.templates.iron.FeII_template_4000_5500.{v}FeII").joinpath("fe_p.txt")    # Group S

        self.fe_in1 = np.loadtxt(str(str1)).T
        self.fe_in2 = np.loadtxt(str(str2)).T
        self.fe_in3 = np.loadtxt(str(str3)).T
        self.fe_in4 = np.loadtxt(str(str4)).T
        self.fe_in5 = np.loadtxt(str(str5)).T

        x_arr_last = np.array( [5700, 10000] )
        y_arr_last = np.array( [0.0, 0.0 ] )

        self.xfe  = self.fe_in1[0]
        self.xfe  = np.concatenate( (self.xfe, x_arr_last))

        self.yfe1 = np.concatenate( (self.fe_in1[1], y_arr_last) )
        self.yfe2 = np.concatenate( (self.fe_in2[1], y_arr_last) )
        self.yfe3 = np.concatenate( (self.fe_in3[1], y_arr_last) )
        self.yfe4 = np.concatenate( (self.fe_in4[1], y_arr_last) )
        self.yfe5 = np.concatenate( (self.fe_in5[1], y_arr_last) )

        self.fe1 = interp1d(self.xfe,self.yfe1,fill_value='extrapolate')
        self.fe2 = interp1d(self.xfe,self.yfe2,fill_value='extrapolate')
        self.fe3 = interp1d(self.xfe,self.yfe3,fill_value='extrapolate')
        self.fe4 = interp1d(self.xfe,self.yfe4,fill_value='extrapolate')
        self.fe5 = interp1d(self.xfe,self.yfe5,fill_value='extrapolate')

    def Fe1(self, x, c):
        model = c* self.fe1(x)
        return model

    def Fe2(self, x, c):
        model = c* self.fe2(x)
        return model        
        
    def Fe3(self, x, c):
        model = c* self.fe3(x)
        return model

    def Fe4(self, x, c):
        model = c* self.fe4(x)
        return model

    def Fe5(self, x, c):
        model = c* self.fe5(x)
        return model

    def feII_temp_option1(self, x, c):
        """ Parameters
        ----------
        x : array-like
            Wavelengths at which to evaluate the FeII template.
        c : float
            Scaling factor for the FeII template.
        Returns
        -------
        All the FeII groups have the same normalization factor.
        """
        ff = self.Fe1(x, 1.) + self.Fe2(x, 1.) + self.Fe3(x, 1.) + self.Fe4(x, 1.) + self.Fe5(x, 1.)
        ff_mean = np.mean(ff)
        return c*(ff/ff_mean)
    
    def feII_temp_option2(self, x, c1, c2, c3, c4, c5):
        """ Parameters
        ----------
        x : array-like
            Wavelengths at which to evaluate the FeII template.
        c1, c2, c3, c4, c5 : float
            Scaling factors for the FeII template groups.
        Returns
        -------
        Each FeII group has its own normalization factor.
        """
        ff = self.Fe1(x, c1) + self.Fe2(x, c2) + self.Fe3(x, c3) + self.Fe4(x, c4) + self.Fe5(x, c5)
        ff_mean = np.mean(ff)
        return (ff/ff_mean)