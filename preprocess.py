#!/usr/bin/env python



# Read modules
import sys
import xarray as xr









'''
code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'
'''




code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'

era5_dir='/fmi/scratch/project_2002138/ERA-5_0p25deg/'



# Read own functions
sys.path.append(code_dir)
import functions as fcts
















# Read ERA5 predictor data
bbox = [59,64, 21,27]

era5_pr = fcts.read_and_select(era5_dir, '', '', era5_dir+'prec*.nc', '', (bbox,),out_resol=False)['domain']
era5_t2 = fcts.read_and_select(era5_dir, '', '', era5_dir+'te2m*.nc', '', (bbox,),out_resol=False)['domain']
era5_mp = fcts.read_and_select(era5_dir, '', '', era5_dir+'pmsl*.nc', '', (bbox,),out_resol=False)['domain']
era5_sm = fcts.read_and_select(era5_dir, '', '', era5_dir+'smo*.nc', '',  (bbox,),out_resol=False)['domain']
era5_sn = fcts.read_and_select(era5_dir, '', '', era5_dir+'snw*.nc', '',  (bbox,),out_resol=False)['domain']

era5_uw = fcts.read_and_select(era5_dir, '', '', era5_dir+'u10m*.nc', '', (bbox,),out_resol=False)['domain']
era5_vw = fcts.read_and_select(era5_dir, '', '', era5_dir+'v10m*.nc', '', (bbox,),out_resol=False)['domain']
era5_tc = fcts.read_and_select(era5_dir, '', '', era5_dir+'tclc*.nc', '', (bbox,),out_resol=False)['domain']
era5_lh = fcts.read_and_select(era5_dir, '', '', era5_dir+'slhf*.nc', '', (bbox,),out_resol=False)['domain']

era5_sh = fcts.read_and_select(era5_dir, '', '', era5_dir+'sshf*.nc', '',  (bbox,),out_resol=False)['domain']
era5_ev = fcts.read_and_select(era5_dir, '', '', era5_dir+'evap*.nc', '',  (bbox,),out_resol=False)['domain']
era5_sw = fcts.read_and_select(era5_dir, '', '', era5_dir+'sswf*.nc', '',  (bbox,),out_resol=False)['domain']
era5_z5 = fcts.read_and_select(era5_dir, '', '', era5_dir+'z150*.nc', '',  (bbox,),out_resol=False)['domain']

era5_st = fcts.read_and_select(era5_dir, '', '', era5_dir+'ste*.nc',  '',  (bbox,),out_resol=False)['domain']
era5_rh = fcts.read_and_select(era5_dir, '', '', era5_dir+'rh1000*.nc','', (bbox,),out_resol=False)['domain']




era5_data = xr.merge([  
                        era5_sh, era5_ev, era5_sw, era5_z5,
                        era5_uw, era5_vw, era5_tc, era5_lh,
                        era5_pr, era5_t2, era5_mp, era5_sm, era5_sn,
                        era5_st, era5_rh
                        ]).sel(time=slice('1996-01-01','2021-12-31')).load()







era5_data.chunk({'time':100}).to_netcdf(data_dir+'era5_preprocessed_0p25deg.nc')



