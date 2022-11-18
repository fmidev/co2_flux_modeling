#!/usr/bin/env python



# Read modules
import sys, ast, importlib, datetime
import numpy as np
import pandas as pd
import xarray as xr




import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

'''
from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
'''







'''
code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'




code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'





'''


code_dir = str(sys.argv[1])+'/'
data_dir = str(sys.argv[2])+'/'
rslt_dir = str(sys.argv[3])+'/'

approach = str(sys.argv[4])








# Read own functions
sys.path.append(code_dir)
import functions as fcts




# Target data 

obs = pd.read_csv(data_dir+'smeardata_20220130_gapfilled.csv')
gap = pd.read_csv(data_dir+'smeardata_20220130_gapfill_method.csv')

obs['gapfill_method'] = gap['HYY_EDDY233.Qc_gapf_NEE']
obs['HYY_EDDY233.NEE'] = obs['HYY_EDDY233.NEE'].where(obs['gapfill_method'] == 0, other=np.nan)

obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

# Resampling to 6H, skip samples containing NaNs
obs = obs[['HYY_EDDY233.NEE']].resample('6H').agg(pd.Series.mean, skipna=False)

vrb = obs.columns[0]



















dropout = ['tcc', 'msl', 'tp', 'swvl1','swvl2','u10','swvl3','v10','r','z','sd','mslhf','e','stl2','stl3','t2m','msdrswrf','stl1'] 
dropout = [] 
#'tcc', 'msl', 'tp', 'swvl1','swvl2','u10','swvl3','v10','r','z','sd','mslhf','e','stl2','stl3','t2m','msdrswrf','stl1'

# Read ERA5 predictor data

dDeg = 0.6; lat_h=61.85; lon_h=24.283
era5_data = xr.open_dataset(data_dir+'era5_preprocessed_0p25deg.nc').sel(
    time=slice('1996-01-01', '2020-12-31'), 
    lat=slice(lat_h - dDeg, lat_h + dDeg), #lat=slice(60,64),
    lon=slice(lon_h - dDeg, lon_h + dDeg), #lon=slice(22,26)
    ).drop(dropout)


vrbs = list(era5_data.data_vars)
vrbs.sort()



'''
for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].plot(); plt.show()



for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].mean(axis=(0)).plot(); plt.show()
'''






# 

# Initial model parameters for Gradient boosting  and Random forest

param_df_gb = pd.read_csv(data_dir+'best_params_Bayes_optim_GB.csv', usecols=['param','value'])
param_df_rf = pd.read_csv(data_dir+'best_params_Bayes_optim_RF.csv', usecols=['param','value'])


params_gb = dict(param_df_gb.values)
params_rf = dict(param_df_rf.values)


params_gb['n_jobs'] = 10
params_gb['num_parallel_tree'] = 1
#params_gb['n_estimators'] = 500



params_rf['n_jobs'] = 10
#params_rf['num_parallel_tree'] = 500
params_rf['n_estimators'] = 1
params_rf['learning_rate'] = 1





'''
    params = {
        'n_jobs': 10,
        'num_parallel_tree': 1, # 1 # 10
        'max_depth': 7,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'colsample_bynode': 1,
        'n_estimators': 200,
        'learning_rate': 0.075 }
    


# Define model parameters for Random forest
if appr == 'rfrs':
    params = {
        'num_parallel_tree': 500, 
        'max_depth': 14,
        'subsample': 0.50,
        'colsample_bytree': 1,
        'colsample_bynode': 0.50,
        'n_estimators': 1 }
    
'''






# 1996–2018
# 1996–2020
all_yrs   = np.arange(1996,2021).astype(int)









t_axis = pd.date_range('1996-01-01', '2019-01-01', freq='6H')
#t_axis = pd.date_range('1996-01-01', '2019-01-01', freq='1H')
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
                Z[vrb].loc[not_nans].values[:,np.newaxis], 'normal',10000)



Z[vrb+'_qt'].loc[not_nans] = transformed.squeeze()





# Prepare predictor matrix X with spatial and temporal lags
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=all_yrs)
X.index = X.index.rename('time'); t_axis = X.index








# Make sure the same time steps are in X and in target Z and they are synchronized
X_time = X.index.values; Z_time = Z.index.values
X_in_Z = np.isin(Z_time, X_time); Z_in_X = np.isin(X_time, Z_time)
X = X.loc[Z_in_X]; Z = Z.loc[X_in_Z]

















# Search for the optimal parameters using Bayesian optimization

if approach == 'GB':
     
    base_estim = fcts.define_xgb(params_gb)
    

    param_grid =    {
                    'learning_rate':         Real(0.01, 0.7, 'log-uniform'),
                    'max_depth':             Integer(3, 18),
                    'n_estimators':          Integer(10,500),
                    'subsample':             Real(0.01, 1.0,  'uniform'),
                    'colsample_bytree':      Real(0.01, 1.0,  'uniform'),
                    'reg_alpha':             Real(1e-9, 1.0,  'log-uniform'),
                    }


if approach == 'RF': 
    
    base_estim = fcts.define_xgb(params_rf)
    
    
    param_grid =    {
                    'max_depth':             Integer(3, 18),
                    'num_parallel_tree':     Integer(10,500),
                    'subsample':             Real(0.01, 1.0,  'uniform'),
                    'colsample_bynode':      Real(0.01, 1.0,  'uniform'), 
                    'reg_alpha':             Real(1e-9, 1.0,  'log-uniform'),
                    }






optimal_ensemble = BayesSearchCV(estimator=base_estim, search_spaces=param_grid, 
                                    n_iter=50, scoring='neg_mean_squared_error', 
                                    cv=KFold(5, shuffle=False), n_points=3, pre_dispatch=4, 
                                    n_jobs=40, return_train_score=True, verbose=99, )


print('Total number of iterations:',optimal_ensemble.total_iterations)

optimal_ensemble.fit(X.loc[not_nans], Z[vrb+'_qt'].loc[not_nans], eval_metric=['rmse', 'rmsle'])




# Save result dataframes
pd.DataFrame(optimal_ensemble.cv_results_).to_csv(data_dir+'cv_results_Bayes_optim_'+approach+'.csv')
pd.DataFrame({  'param':list(optimal_ensemble.best_params_.keys()), 
                'value':list(optimal_ensemble.best_params_.values())}).to_csv(data_dir+'best_params_Bayes_optim_'+approach+'.csv')








