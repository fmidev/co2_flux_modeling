#!/usr/bin/env python



# Read modules
import sys, ast, importlib, datetime
import numpy as np
import pandas as pd
import xarray as xr




import xgboost as xgb



import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import resample


import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature





code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'







# Read own functions
sys.path.append(code_dir)
import functions as fcts









# Target data 
obs = pd.read_csv(data_dir+'smeardata_20210224_set1.csv')
obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

# Resampling to 6H, skip samples containing NaNs
obs = obs[['HYY_EDDY233.F_c']].resample('6H').agg(pd.Series.mean, skipna=False)

vrb = obs.columns[0]

























#bbox = [ 61.84741, 24.29477]

# Read ERA5 predictor data
era5_data = xr.open_dataset(data_dir+'era5_preprocessed.nc').sel(
    time=slice('1996-01-01', '2018-12-31'), 
    lat=slice(60,64),
    lon=slice(22,26)
    )#.drop(['tcc','msl','swvl2','swvl1','tp','sd','swvl3','v10','u10','z','mslhf','r','e','stl3','stl2','t2m','msdrswrf','stl1'])


vrbs = list(era5_data.data_vars)
vrbs.sort()



'''
for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].plot(); plt.show()



for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].mean(axis=(0)).plot(); plt.show()
'''








# Define parameter space for tuning
num_parallel_tree = 10 # 5 # 10
max_depth = 7
subsample=0.75
colsample_bytree=0.75
early_stopping = True
nstm = 500

step = 0
wndw = 1






# 1996–2018
all_yrs   = np.arange(1996,2019).astype(int)








models  = []
vld_yrs = []


t_axis = pd.date_range('1996-01-01', '2019-01-01', freq='6H')
Z = pd.DataFrame(index=t_axis)

Z[obs.columns[0]] = obs.copy(deep=True)


Z[vrb+'_qt'] = np.nan 



# Flag suspicious values (deviations larger than 7*std)
nans = np.zeros(Z[vrb].values.shape, 'bool')

std = np.nanstd(Z[vrb])
nans[np.abs(Z[vrb])>std*7] = True
Z[vrb].loc[np.abs(Z[vrb])>std*7] = np.nan

nans[np.isnan(Z[vrb].values)] = True
not_nans = ~nans 


# Define and apply the quantile model for predictand
transformed, qt_model = fcts.transform_to_normal_distribution(
                Z[vrb].loc[not_nans].values[:,np.newaxis], 'normal',100000)



Z[vrb+'_qt'].loc[not_nans] = transformed.squeeze()



# Additional in-sample sampling to study the sensitivity of the model on the size of fitting data
sample_sizes = np.arange(0.1,1.01,0.1)

for ss in sample_sizes:
    Z['fcs_'+str(int(ss*100))]  = np.nan







# Prepare predictor matrix X 
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=np.arange(1996,2019))
X.index = X.index.rename('time'); t_axis = X.index


# Shuffle X columns for good measure
import random; random.seed(99)
columns = list(X.columns)
random.shuffle(columns)

X = X[columns]












# Make sure the same time steps are in X and in target Z and they are synchronized
X_time = X.index.values; Z_time = Z.index.values
X_in_Z = np.isin(Z_time, X_time); Z_in_X = np.isin(X_time, Z_time)
X = X.loc[Z_in_X]; Z = Z.loc[X_in_Z]






# Fit models in cross-validation framework
kf = KFold(5, shuffle=True)
for trn_idx, tst_idx in kf.split(all_yrs):
    tst_yrs = all_yrs[tst_idx]
    trn_yrs = all_yrs[trn_idx]
    print(trn_yrs, tst_yrs)
    vld_yrs.append(tst_yrs)
    
    t = fcts.stopwatch('start')
    fcts=importlib.reload(fcts)
    
    
    # Define indices for train and test sets
    fit_idx_y = np.isin(Z.index.year,  trn_yrs) 
    fit_idx_x = np.isin(X.index.year,  trn_yrs)  
    
    tst_idx_y = np.isin(Z.index.year,  tst_yrs) 
    tst_idx_x = np.isin(X.index.year,  tst_yrs)  
    
    
    # Make sure the same time steps are in X and Y and they are synchronized
    t_Z_tst = Z[vrb][tst_idx_y].dropna().index
    t_Z_fit = Z[vrb][fit_idx_y].dropna().index
    
    t_X_tst = X[tst_idx_x].index
    t_X_fit = X[fit_idx_x].index
    
    t_common_fit = np.intersect1d(t_Z_fit, t_X_fit)
    t_common_tst = np.intersect1d(t_Z_tst, t_X_tst)
    
    for ss in sample_sizes:
        # Define the model
        base_estim = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=nstm,  
            learning_rate=0.075, 
            max_depth=max_depth,
            alpha=0.01,
            num_parallel_tree=num_parallel_tree,
            n_jobs=40, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,  
            random_state=99)
        
        sample_size = int(len(X.loc[t_common_fit])*ss)
        t_fit_sampled = resample(t_common_fit, replace=False, n_samples=sample_size)
        
        # Train
        verbose=True
        fitted_ensemble, = fcts.fit_ensemble(
            X.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], 
            X.loc[t_common_tst], Z[vrb+'_qt'].loc[t_common_tst], 
            base_estim, 'rmse', verbose=True, early_stopping=early_stopping)
        
        
        if ss==1.0:
            # Save the model if using full 100% sample size
            models.append(fitted_ensemble)
        
        best_nstm = fitted_ensemble.best_ntree_limit
        
        print('Fitting',vrb,best_nstm,step,wndw,'took ' + fcts.stopwatch('stop', t))
        
        # Save the inverse transformed forecast
        prediction = fitted_ensemble.predict(X.iloc[tst_idx_x].values)[:,np.newaxis]
        Z['fcs_'+str(int(ss*100))].iloc[tst_idx_y] = qt_model.inverse_transform(prediction).squeeze() 
        




# Save modeled results and models
Z.drop(columns=[vrb+'_qt']).to_csv(data_dir+'obs_mod_co2.csv')

for i,mdl in enumerate(models):
    mdl.save_model(data_dir+'model_'+str(i+1)+'_for_co2.json')








# -                                                                                     CORR = 0.951 RMSE = 1.096 CORR-w = 0.970 RMSE-w = 0.387
# - tcc:                                                                                CORR = 0.951 RMSE = 1.104 CORR-w = 0.969 RMSE-w = 0.390
# - tcc,msl:                                                                            CORR = 0.951 RMSE = 1.100 CORR-w = 0.970 RMSE-w = 0.384
# - tcc,msl,swvl2:                                                                      CORR = 0.951 RMSE = 1.101 CORR-w = 0.970 RMSE-w = 0.384
# - tcc,msl,swvl2,swvl1:                                                                CORR = 0.951 RMSE = 1.101 CORR-w = 0.971 RMSE-w = 0.379
# - tcc,msl,swvl2,swvl1,tp:                                                             CORR = 0.950 RMSE = 1.110 CORR-w = 0.969 RMSE-w = 0.390
# - tcc,msl,swvl2,swvl1,tp,sd:                                                          CORR = 0.950 RMSE = 1.115 CORR-w = 0.968 RMSE-w = 0.395
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3:                                                    CORR = 0.951 RMSE = 1.102 CORR-w = 0.969 RMSE-w = 0.388
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10:                                                CORR = 0.950 RMSE = 1.105 CORR-w = 0.970 RMSE-w = 0.386
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10:                                            CORR = 0.951 RMSE = 1.101 CORR-w = 0.972 RMSE-w = 0.372
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z:                                          CORR = 0.950 RMSE = 1.113 CORR-w = 0.970 RMSE-w = 0.385
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf:                                    CORR = 0.950 RMSE = 1.115 CORR-w = 0.970 RMSE-w = 0.383

# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r:                                  CORR = 0.946 RMSE = 1.149 CORR-w = 0.970 RMSE-w = 0.384
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e:                                CORR = 0.943 RMSE = 1.182 CORR-w = 0.967 RMSE-w = 0.400
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e,stl3:                           CORR = 0.944 RMSE = 1.177 CORR-w = 0.968 RMSE-w = 0.394
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e,stl3,stl2:                      CORR = 0.943 RMSE = 1.187 CORR-w = 0.966 RMSE-w = 0.405
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e,stl3,stl2,t2m:                  CORR = 0.942 RMSE = 1.196 CORR-w = 0.965 RMSE-w = 0.411
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e,stl3,stl2,t2m,msdrswrf:         CORR = 0.936 RMSE = 1.254 CORR-w = 0.962 RMSE-w = 0.427
# - tcc,msl,swvl2,swvl1,tp,sd,swvl3,v10,u10,z,mslhf,r,e,stl3,stl2,t2m,msdrswrf,stl1:    CORR = 0.899 RMSE = 1.557 CORR-w = 0.943 RMSE-w = 0.521




