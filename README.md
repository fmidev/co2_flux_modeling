# Modeling the CO2 flux in Hyytiälä with gradient boosting

This code reproduces the data and experiments presented in 

`Kämäräinen et al.: Modeling the sub-daily variability of the atmosphere–forest CO2 flux with machine learning`

## System requirements
The code was developed in the UNIX/Linux environment using Python 3.7.6.

Running the code takes 2–4 hours with 40 CPU cores and ~ 100Gb RAM.
Running it should be possible with considerably smaller resources, but
the run time will increase accordingly. To ease the computing, change
the `num_parallel_tree` parameter from 10 to 1 in fit_optim.py. Be aware 
that this will change the results to some extent.

## Dependencies
Several external Python libraries were installed with the Miniconda installer, available here:
https://docs.conda.io/en/latest/miniconda.html

The conda-forge repository was used to install the libraries:
`conda install -c conda-forge python=3.7.6 numpy=1.18.1 scipy=1.4.1 matplotlib=3.1.2 
    xarray=0.13.0 netCDF4=1.5.3 bottleneck=1.3.0 dask=2.8.0 seaborn=0.11.1 pandas=0.25.3 
    scikit-learn=0.21.3 xgboost=1.2.0`

For downloading the ERA5 data the CDS API client needs to be installed with pip:
`pip install cdsapi`

Also, follow these instructions to set up the CDS API key to your home folder:
https://cds.climate.copernicus.eu/api-how-to

## Downloading the input data  
For downloading the ERA5 data, run the run_download_era5.sh Unix shell file. Prior to running 
the file, make sure that your miniconda environment is actvated, that the folder structures 
are correctly defined inside the files, and that all Python dependencies are installed. 

Running the file in Unix:
`./run_download_and_preprocess_data.sh`

## Running the experiments and analyses
Unzip the CO2 observations (smeardata_20210224_set1.zip) and place the file in a suitable location,
for example in the same folder where the code is.

Make sure the folder paths are correctly defined inside the Python scripts prior to running them.  

First run the preprocessing script:
`python preprocess.py`

Then the fitting part:
`python fit_optim.py`

Finally the analysis and plotting:
`python analyse_results.py`
