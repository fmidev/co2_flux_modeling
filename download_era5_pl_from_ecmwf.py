#!/usr/bin/env python
#
#

import os, sys, itertools
import numpy as np
import cdsapi
server = cdsapi.Client()



name2code = {
             'z10':     ['10',    'geopotential'],
             'z50':     ['50',    'geopotential'],
             'z150':    ['150',   'geopotential'],
             'z500':    ['500',   'geopotential'],
             'u200':    ['200',   'u_component_of_wind'],
             'u850':    ['850',   'u_component_of_wind'],
             't850':    ['850',   'temperature'],
             'rh1000':  ['1000',  'relative_humidity'],
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
                'area'          : [65, 15, 55, 32], #[70, 15, 45, 40],
                'variable'      : name2code[varname][1],
                'pressure_level': name2code[varname][0],
                'year'          : '%04d' % (year),
                'month'         : ['%02d' % (i+1) for i in range(12)], 
                'day'           : ['%02d' % (i+1) for i in range(31)], 
                'time'          : ['%02d:00' % (i*6) for i in range(4)], 
                'format'        : 'netcdf',
               }
        
        print('Fetching data for', nc_file)
        
        try:
            # 'reanalysis-era5-pressure-levels-preliminary-back-extension'
            # 'reanalysis-era5-pressure-levels'
            server.retrieve('reanalysis-era5-pressure-levels', opts, nc_file) 
        except:
            print('Retrieval failed for',nc_file)
            



