# AGNoptspecNN

**Version:** 0.1.0 (Initial Release)

A spectral classification package using convolutional neural networks to classify AGNs into Type 1, Type 1.9, and Type 2 based on optical spectra.

## Installation
This package is not still available at PyPI, so an editable install can be made locally in your system.

```bash
pip install -e .
```

**Dependencies**
* [numpy](https://numpy.org)
* [scipy](https://scipy.org/)
* [astropy](https://www.astropy.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [tensorflow](https://www.tensorflow.org/)

## Disclaimer
The classification might not always be correct, as the training sample can have limitations. This will improve over the course of time as we upgrade the data simulation model and incorporate real optical spectra in the training sample.

## Upcoming Upgrades
* The package now uses simple simulated data for classification. The simulation models will be updated, and observed data will also be used for training.
* Generalization to classify all extragalactic astrophysical spectra: galaxies, AGNs, TDEs.
* Incorporate real spectra into the training data.
* Implementation of Bayesian Neural Networks.
* Integrate data upload via Zenodo API.

## WebApp
Please check out our webapp [here](https://agnoptspecnn-web-dwznx4k3b8be8v2r2emupl.streamlit.app/). The repository of the webapp can be found here: [https://github.com/tathagatas1996/AGNoptspecNN-web](https://github.com/tathagatas1996/AGNoptspecNN-web)


## Acknowledgements:
The package make use of the galaxy templates from [Mannucci et al. (2001), MNRAS, 326, 745](https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..745M/abstract) and the iron FeII templates from http://servo.aob.rs/FeII_AGN/link3.html.