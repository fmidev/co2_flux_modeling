#!/usr/bin/env python
#
#

import os, sys, itertools
import numpy as np
import cdsapi
server = cdsapi.Client()




             


name2code = {
                'pmsl':     'mean_sea_level_pressure',
                'te2m':     '2m_temperature',
                'sst':      'sea_surface_temperature',
                'snw':      'snow_depth',
                'sic':      'sea_ice_cover',
                'smo':      ['volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3'],
                'ste':      ['soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',],
                'prec':     'total_precipitation',
                'lsmask':   'land_sea_mask',
                'u10m':     '10m_u_component_of_wind',
                'v10m':     '10m_v_component_of_wind',
                'wg10':     'instantaneous_10m_wind_gust',
                'evap':     'evaporation',
                'sswf':     'mean_surface_direct_short_wave_radiation_flux',
                'slhf':     'mean_surface_latent_heat_flux',
                'sshf':     'mean_surface_sensible_heat_flux',
                'tclc':     'total_cloud_cover',
            }


varname = str(sys.argv[1]) 


years = np.arange(1995,2022).astype(int)


for year in years:
    basename = '%s_era5_1p0deg_%04d' % (varname,year)
    nc_file  = '%s.nc'  % (basename)
    
    if os.path.exists(nc_file):
        pass
    
    else:
        
        opts = {
                'product_type'  : 'reanalysis',  
                #'grid'          : '1/1',
                'area'          : [66, 15, 55, 32], #[70, 15, 45, 40],
                'variable'      : name2code[varname], 
                'year'          : '%04d' % (year),
                'month'         : ['%02d' % (i+1) for i in range(12)], 
                'day'           : ['%02d' % (i+1) for i in range(31)], 
                'time'          : ['%02d:00' % (i*6) for i in range(4)], 
                'format'        : 'netcdf',
               }
        
        print('Fetching data for', nc_file)
        
        try:
            # 'reanalysis-era5-single-levels-preliminary-back-extension'
            # 'reanalysis-era5-single-levels'
            server.retrieve('reanalysis-era5-single-levels', opts, nc_file)
        
        except:
            print('Retrieval failed for',nc_file)
            



