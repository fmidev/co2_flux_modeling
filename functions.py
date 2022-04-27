#!/usr/bin/env python



import itertools
import numpy as np
import pandas as pd
import xarray as xr












# --- Data analysis and evaluation ---

def calc_bootstrap(obs,fcs,func, bootstrap_range, B):
    from sklearn.utils import resample
    
    idxs = np.arange(len(fcs))
    results = []

    random_state = 0
    for smp in range(B):
        random_state += 1
        sample = resample(idxs, replace=True, n_samples=len(fcs), random_state=random_state)
        results.append(func(obs[sample], fcs[sample]))
    
    return np.percentile(results, bootstrap_range), results



def calc_rmse(a,b,axis=0): 
    return np.sqrt(np.nanmean((a-b)**2, axis=axis))

def calc_mse(a,b,axis=0):  
    return np.nanmean((a-b)**2, axis=axis)
    


def calc_corr(a, b, axis=0):
    mask_a = np.isnan(a) | np.isinf(a)
    mask_b = np.isnan(b) | np.isinf(b)
    mask = mask_a + mask_b
    _a = a.copy()
    _b = b.copy()
    try:
        _a[mask] = np.nan
        _b[mask] = np.nan
    except:
        pass
    _a = _a - np.nanmean(_a, axis=axis, keepdims=True)
    _b = _b - np.nanmean(_b, axis=axis, keepdims=True)
    std_a = np.sqrt(np.nanmean(_a**2, axis=axis)) 
    std_b = np.sqrt(np.nanmean(_b**2, axis=axis)) 
    return np.nanmean(_a * _b, axis=axis)/(std_a*std_b)



# --- Manipulators ---



def kz_filter(da, window, iterations, center=True):
    """
    KZ filter implementation for xarray/pandas
    Window is the filter window m in the units of the data (m = 2q+1)
    Iterations is the number of times the moving average is evaluated
    """
    import xarray as xr
    import pandas as pd
    
    for i in range(iterations):
        if type(da) == xr.core.dataarray.DataArray:
            da = da.rolling(time=window, min_periods=1, center=center).mean()
        
        if type(da) == pd.core.frame.DataFrame or type(da) == pd.core.series.Series:
            da = da.rolling(window=window, min_periods=1, center=center).mean()
    
    return da



    

def transform_to_normal_distribution(data, output_dist='normal', nq=1000):
    from sklearn.preprocessing import QuantileTransformer
    
    qt = QuantileTransformer(n_quantiles=nq, output_distribution=output_dist, random_state=99)
    transformed = qt.fit_transform(data)
    return transformed, qt










# --- Reading and data extraction ---

def bool_index_to_int_index(bool_index):
    return np.where(bool_index)[0]



def combine_and_define_lags(ds, lags, all_yrs):
    
    
    data_out = []
    for name_out in list(ds.data_vars):
        print('Handling',name_out, ds[name_out].shape)
        
        if len(ds[name_out].shape) == 3:
            print('Stacking dataset')
            grd_vls = ds[name_out].stack(gridcell=('lat', 'lon'))
            n_gcl = grd_vls.shape[1]
        
        if len(ds[name_out].shape) == 1:
            print('Skip stacking')
            grd_vls = ds[name_out]
            n_gcl = 1
        
        grd_vls = grd_vls.where(~np.isinf(grd_vls), other=0)        
        
        time = grd_vls.time
        # Indexes 
        all_idx = bool_index_to_int_index(np.isin(grd_vls['time.year'], all_yrs))
        
        if len(data_out) == 0:
            data_out = pd.DataFrame(index=time.values[all_idx]) 
        
        print(grd_vls[all_idx].shape)
        data_values = grd_vls[all_idx]
        
        
        data = np.zeros((len(time.values[all_idx]), len(lags)*n_gcl))
        
        # Lags and Lead times
        i=0; columns = []
        for lag in lags:
            
            sign = '+'
            if np.sign(lag)==1: sign = '-'
            
            
            for gcl in range(n_gcl):
                output_vrb = 'Xv_'+name_out+'_'+str(gcl+1)+sign+str(np.abs(lag))
                columns.append(output_vrb)
                try:
                    data[:,i] = np.roll(data_values[:,gcl], lag) 
                except:
                    data[:,i] = np.roll(data_values[:], lag) 
                
                if sign=='-':
                    data[0:np.abs(lag),i] = np.nan
                
                if sign=='+':
                    data[-lag:0,i] = np.nan
                
                i+=1
        
        
        df = pd.DataFrame(data=data, index=time.values[all_idx], columns=columns)
        data_out = pd.concat([data_out, df,], join='outer', axis=1)
    
    # Fill (potential) holes  
    data_out = data_out.ffill().bfill()
    
    return data_out








def adjust_lats_lons(ds):
    coord_names =   [
                    ['longitude', 'latitude'],
                    ['X', 'Y'],
                    ]
    for nmes in coord_names:
        try:
            ds = ds.rename({nmes[0]: 'lon', nmes[1]: 'lat'}) 
        except: 
            pass  
    
    
    if(ds.lon.values.max() > 180):
        print('Transforming longitudes from [0,360] to [-180,180]', flush=True)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        
    return ds.sortby(['lon','lat'])
    


def read_and_select(in__dir, name_out,source, fles, var, domains,out_resol, mask_seas=False, smooth_data=False):
    ''' 
    (1) Transform longitudes from [0,360] to [-180,180], 
    (2) reverse latitudes (if needed), 
    (3) and select area(s) of interest
    '''
    
    try:
        ds = xr.open_mfdataset(fles, parallel=True, combine='by_coords', chunks=len(fles), engine='netcdf4')
        print('Combined files by coords')
    except:
        ds = xr.open_mfdataset(fles, parallel=True, combine='nested', concat_dim='time', engine='netcdf4')
        print('Combined files by nesting')

    if np.isin('expver', list(ds.dims)):
        print('expver',var,ds.expver)
        import matplotlib.pyplot as plt
        try:
            for ev in ds.expver.values:
                ds[var].sel(expver=ev).mean(axis=[1,2]).plot();plt.show() 
         
        except: pass
        
        ds = ds.sel(expver=ds.expver.values.min())
    
    ds = ds.squeeze()
    
    if out_resol:
        # Interpolate all data to the defined resolution
        lon = np.arange(-180,180,out_resol)
        lat = np.arange(-90,90,out_resol)
        ds = ds.interp(lon=lon, lat=lat)
    
    # Ensure same coordinates in all datasets
    ds = adjust_lats_lons(ds) #.load())
    
    if smooth_data:
        sigma3d = smooth_data
        ds = ds.apply(Gauss_filter, sigma3d)
    
    if mask_seas:
        # Land-sea-mask
        lsm = xr.open_dataset(mask_seas).mask[0].drop('time')
        lsm = adjust_lats_lons(lsm.assign_coords(lon=np.floor(lsm.lon.values), 
                                                 lat=np.floor(lsm.lat.values)))
        
        if out_resol:
            lon = np.arange(-180,180,out_resol)
            lat = np.arange(-90,90,out_resol)
            lsm = lsm.interp(lon=lon, lat=lat, method='nearest')#.load()
        
        ds[var] = ds[var].where(lsm == 1, other=np.nan)
    
    data_out = {}
    for domain in domains:
        
        print(domain, flush=True)
        
        
        if len(domain) == 4:
            data_out['domain'] = ds.sel(
                lat=slice(domain[0],domain[1]), 
                lon=slice(domain[2],domain[3]))#.load() #.where(mask, other=np.nan)

        elif len(domain) == 2:
            data_out['domain'] = ds.sel(method='nearest',
                lat=domain[0], lon=domain[1])#.load() #.where(mask, other=np.nan)
    
        else: print('Check domain definition!!!')
    
    return data_out 










# --- Model fitting ---


def fit_ensemble(X_trn, Y_trn, X_val, Y_val, base_estim, 
                    eval_metric,verbose=True,early_stopping=True):
    
    try: X_trn, X_val = X_trn.values, X_val.values
    except: pass
    
    try: Y_trn, Y_val = Y_trn.values, Y_val.values
    except: pass
    
    
    import xgboost as xgb
    
    if early_stopping:
        fitting = base_estim.fit(X_trn, Y_trn,
            eval_set=[  (X_trn, Y_trn), 
                        (X_val, Y_val)],
            eval_metric=eval_metric,
            early_stopping_rounds=200,
            verbose=verbose)
    else:
        fitting = base_estim.fit(X_trn, Y_trn,
            eval_set=[  (X_trn, Y_trn), 
                        (X_val, Y_val)],
            eval_metric=eval_metric,
            verbose=verbose)

    
    
    
    return fitting






# --- Misc ---


def stopwatch(start_stop, t=-99):
    import time
    
    if start_stop=='start':
        t = time.time()
        return t
    
    if start_stop=='stop':
        elapsed = time.time() - t
        return time.strftime("%H hours, %M minutes, %S seconds",time.gmtime(elapsed))


