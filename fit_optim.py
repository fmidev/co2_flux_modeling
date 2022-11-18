#!/usr/bin/env python



# Read modules
import sys, ast, importlib, datetime, glob, itertools
import numpy as np
import pandas as pd
import xarray as xr




import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.utils import resample
from sklearn.model_selection import train_test_split









'''



code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'


code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'
approach='GB'
shuffle=True



'''

code_dir = str(sys.argv[1])+'/'
data_dir = str(sys.argv[2])+'/'
rslt_dir = str(sys.argv[3])+'/'
approach = str(sys.argv[4])

# Whether to shuffle the additional in-sample samples (sub-samples)
shuffle  = ast.literal_eval(sys.argv[5])

# Whether to use standard or repeated KFold cross-validation
repeat = True
n_repeats = 8

# File identifier
idnt = approach+'_shuffle='+str(shuffle)





# Read own functions
sys.path.append(code_dir)
import functions as fcts



# Optimized/Initial model parameters for Gradient boosting or Random forest
param_df = pd.read_csv(data_dir+'best_params_Bayes_optim_1_'+idnt+'.csv', usecols=['param','value'])
params = dict(param_df.values)



params['n_jobs'] = 40
params['max_depth'] = int(params['max_depth'])





if approach == 'GB':
    params['num_parallel_tree'] = 1
    params['n_estimators'] = 500
    
    # Define the search space for Bayesian optimization
    param_grid =    {
                    'learning_rate':         Real(0.01, 0.7, 'log-uniform'),
                    'max_depth':             Integer(3, 18),
                    'n_estimators':          Integer(10,1000),
                    'subsample':             Real(0.01, 1.0,  'uniform'),
                    'colsample_bytree':      Real(0.01, 1.0,  'uniform'),
                    'reg_alpha':             Real(1e-9, 1.0,  'log-uniform'),
                    }



if approach == 'RF':
    params['num_parallel_tree'] = 500
    params['n_estimators'] = 1
    params['learning_rate'] = 1
    
    # Define the search space for Bayesian optimization
    param_grid =    {
                    'max_depth':             Integer(3, 18),
                    'num_parallel_tree':     Integer(10,1000),
                    'subsample':             Real(0.01, 1.0,  'uniform'),
                    'colsample_bynode':      Real(0.01, 1.0,  'uniform'), 
                    'reg_alpha':             Real(1e-9, 1.0,  'log-uniform'),
                    }
















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




















dropout = ['z','swvl2','msl','sd','swvl3','swvl1','tcc','stl2','tp','stl3','mslhf','v10','u10','stl1','r','msdrswrf','e','msshf','t2m',] 
dropout = ['z','swvl2','msl','sd','swvl3','swvl1','tcc','stl2','tp'] 
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







# 

# Define model parameters for Gradient boosting
if appr == 'gbst':
    param_df = pd.read_csv(data_dir+'best_params_Bayes_optim_10.csv', usecols=['param','value'])
    params = dict(param_df.values)
    
    
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



# Additional in-sample sampling to study the sensitivity of the model on the size of fitting data
sample_sizes = np.arange(0.1,1.01,0.1)
#sample_sizes = [1.0]


for ss in sample_sizes:
    Z['fcs_0_'+str(int(ss*100))]  = np.nan
    Z['ctr_0_'+str(int(ss*100))]  = np.nan


if repeat:
    for rpt in range(n_repeats):
        for ss in sample_sizes:
            Z['fcs_'+str(rpt)+'_'+str(int(ss*100))]  = np.nan
            Z['ctr_'+str(rpt)+'_'+str(int(ss*100))]  = np.nan




# Prepare predictor matrix X with spatial and temporal lags
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=all_yrs)
X.index = X.index.rename('time'); t_axis = X.index


# Predictor data for a control experiment WITHOUT spatial or temporal lags
C    = fcts.combine_and_define_lags(era5_data.sel(lat=61.85, lon=24.283, method='nearest'), [0], all_yrs=all_yrs)
C.index = C.index.rename('time'); t_axis = C.index







# Make sure the same time steps are in X and in target Z and they are synchronized
X_time = X.index.values; Z_time = Z.index.values
X_in_Z = np.isin(Z_time, X_time); Z_in_X = np.isin(X_time, Z_time)
X = X.loc[Z_in_X]; C = C.loc[Z_in_X]; Z = Z.loc[X_in_Z]







# Fit models in cross-validation framework



#models_full_model  = []; models_ctrl_model  = []
#vld_yrs = []


if repeat:
    kf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=99)
else:
    kf = KFold(n_splits=5, shuffle=True, random_state=99) 



fold = 0
split_count=0
for trn_idx, tst_idx in kf.split(all_yrs):
    
    if repeat:
        fold = np.mod(split_count,5) + 1
        rept = np.floor_divide(split_count,5) 
    else:
        fold += 1
        rept = 0
    
    tst_yrs = all_yrs[tst_idx]
    trn_yrs = all_yrs[trn_idx]
    print(fold,trn_yrs, tst_yrs)
    #vld_yrs.append(tst_yrs)
    
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
        
        #sample_size = int(len(X.loc[t_common_fit])*ss)
        #t_fit_sampled = resample(t_common_fit, replace=False, n_samples=sample_size)
        if ss==1.0:
            smpl=0.999999999
            optimize=True
        else:
            smpl=ss
            optimize=False
        
        t_fit_sampled, _ = train_test_split(t_common_fit, train_size=smpl, shuffle=shuffle)
        
        '''
        # Optimize and fit full model
        params['n_jobs'] = 10
        fitted_ensemble, opt = fcts.fit_optimize(approach, params, param_grid, 
                                        X.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], optimize=optimize)
        
        # Fit control model
        if optimize:
            opt_params = opt.best_params_
        else:
            opt_params = params
        
        opt_params['n_jobs'] = 40
        contrl_ensemble, _ = fcts.fit_optimize(approach, opt_params, param_grid, 
                                        C.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], optimize=False)
        '''
        
        #'''
        
        
        # Refit
        opt_params = pd.read_csv(data_dir+'best_params_Bayes_optim_'+str(fold)+'_'+idnt+'.csv', usecols=['param','value'])
        opt_params = dict(param_df.values)
        opt_params['max_depth'] = int(opt_params['max_depth'])
        opt_params['n_jobs'] = 40
        
        fitted_ensemble, _ = fcts.fit_optimize(approach, opt_params, param_grid, 
                                        X.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], optimize=False)        
        contrl_ensemble, _ = fcts.fit_optimize(approach, opt_params, param_grid, 
                                        C.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], optimize=False)
        
        
        #'''
        
        #'''
        if ss==1.0:
            print(ss)
            
            # Save the optimized model and parameters if using full 100% sample size
            contrl_ensemble.save_model(data_dir+'cntrl_'+str(rept)+'_'+str(fold)+'_'+idnt+'.json')
            fitted_ensemble.save_model(data_dir+'model_'+str(rept)+'_'+str(fold)+'_'+idnt+'.json')
            '''
            cv_reslts = pd.DataFrame(opt.cv_results_)
            opt_parms = pd.DataFrame({  'param':list(opt.best_params_.keys()), 
                                        'value':list(opt.best_params_.values())})
            
            cv_reslts.to_csv(data_dir+'cv_results_Bayes_optim_'+str(rept)+'_'+str(fold)+'_'+idnt+'.csv')
            opt_parms.to_csv(data_dir+'best_params_Bayes_optim_'+str(rept)+'_'+str(fold)+'_'+idnt+'.csv')
            '''
            
        #'''
        print('Fitting and optimizing',approach,fold,ss,vrb,'took ' + fcts.stopwatch('stop', t))
        print(dropout)
        
        # Save the inverse transformed forecast
        full_model_prediction = fitted_ensemble.predict(X.iloc[tst_idx_x].values)[:,np.newaxis]
        ctrl_model_prediction = contrl_ensemble.predict(C.iloc[tst_idx_x].values)[:,np.newaxis]
        
        Z['fcs_'+str(rept)+'_'+str(int(ss*100))].iloc[tst_idx_y] = qt_model.inverse_transform(full_model_prediction).squeeze() 
        Z['ctr_'+str(rept)+'_'+str(int(ss*100))].iloc[tst_idx_y] = qt_model.inverse_transform(ctrl_model_prediction).squeeze() 
    
    
    
    split_count += 1





not_nans = ~ np.isnan(Z[vrb]).values


for rpt,d in itertools.product(range(n_repeats), ['fcs','ctr']):
    cor_6h = fcts.calc_corr(Z[d+'_'+str(rpt)+'_100'].loc[not_nans], Z[vrb].loc[not_nans])
    rms_6h = fcts.calc_rmse(Z[d+'_'+str(rpt)+'_100'].loc[not_nans], Z[vrb].loc[not_nans])
    nse_6h = fcts.calc_nses(Z[d+'_'+str(rpt)+'_100'].loc[not_nans], Z[vrb].loc[not_nans])
    r2s_6h = fcts.calc_r2ss(Z[d+'_'+str(rpt)+'_100'].loc[not_nans], Z[vrb].loc[not_nans])
    
    print(dropout)
    print(  str(rpt)+' '+d+  ': cor_6h = '+str(cor_6h.round(4))+\
                ' rms_6h = '+str(rms_6h.round(4))+\
                ' nse_6h = '+str(nse_6h.round(4))+\
                ' r2s_6h = '+str(r2s_6h.round(4)))



'''
fcs: cor_6h = 0.9545342 rms_6h = 1.2183495 nse_6h = 0.9056311 r2s_6h = [0.9056265]
fcs: cor_1w = 0.9694754 rms_1w = 0.7801227 nse_1w = 0.9390132 r2s_1w = [0.9390129]
ctr: cor_6h = 0.8515796 rms_6h = 2.2082477 nse_6h = 0.4915807 r2s_6h = [0.4837893]
ctr: cor_1w = 0.928645  rms_1w = 1.2732213 nse_1w = 0.774461  r2s_1w = [0.7689431]
'''




'''
ctrl_model_prediction = contrl_ensemble.predict(C.iloc[tst_idx_x], output_margin=True)[:,np.newaxis]

explainer = shap.TreeExplainer(contrl_ensemble)
shap_values = explainer.shap_values(C.iloc[tst_idx_x])
np.abs(shap_values.sum(1) + explainer.expected_value - ctrl_model_prediction).max()

shap.summary_plot(shap_values, C.iloc[tst_idx_x], plot_type='violin')
shap.summary_plot(shap_values, C.iloc[tst_idx_x], plot_type='bar')

#shap.force_plot(explainer.expected_value, shap_values, C.iloc[tst_idx_x])

shap.plots.bar(shap_values, C.iloc[tst_idx_x])


shap.plots.bar(shap_values[0])
'''




# CORR = 0.955 RMSE = 1.217 CORR-w = 0.969 RMSE-w = 0.788



# Save modeled results and models
Z.drop(columns=[vrb+'_qt']).to_csv(data_dir+'obs_mod_co2_'+idnt+'.csv')




# New Dropout experiment
# -                                                                                     fcs: cor_6h = 0.9540 rms_6h = 1.2255 nse_6h = 0.9041 r2s_6h = 0.9041
# - z:                                                                                  fcs: cor_6h = 0.9535 rms_6h = 1.2322 nse_6h = 0.9029 r2s_6h = 0.9029
# - z,swvl2:                                                                            fcs: cor_6h = 0.9539 rms_6h = 1.2262 nse_6h = 0.9040 r2s_6h = 0.9040
# - z,swvl2,msl:                                                                        fcs: cor_6h = 0.9540 rms_6h = 1.2258 nse_6h = 0.9045 r2s_6h = 0.9045
# - z,swvl2,msl,sd:                                                                     fcs: cor_6h = 0.9540 rms_6h = 1.2257 nse_6h = 0.9038 r2s_6h = 0.9038
# - z,swvl2,msl,sd,swvl3:                                                               fcs: cor_6h = 0.9543 rms_6h = 1.2220 nse_6h = 0.9053 r2s_6h = 0.9053
# - z,swvl2,msl,sd,swvl3,swvl1:                                                         fcs: cor_6h = 0.9547 rms_6h = 1.2166 nse_6h = 0.9066 r2s_6h = 0.9066
# - z,swvl2,msl,sd,swvl3,swvl1,tcc:                                                     fcs: cor_6h = 0.9555 rms_6h = 1.2070 nse_6h = 0.9080 r2s_6h = 0.9080

# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2:                                                fcs: cor_6h = 0.9550 rms_6h = 1.2129 nse_6h = 0.9070 r2s_6h = 0.9070
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp:                                             fcs: cor_6h = 0.9551 rms_6h = 1.2120 nse_6h = 0.9076 r2s_6h = 0.9076
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3:                                        fcs: cor_6h = 0.9550 rms_6h = 1.2142 nse_6h = 0.9074 r2s_6h = 0.9074
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf:                                  fcs: cor_6h = 0.9550 rms_6h = 1.2136 nse_6h = 0.9073 r2s_6h = 0.9073
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10:                              fcs: cor_6h = 0.9540 rms_6h = 1.2261 nse_6h = 0.9052 r2s_6h = 0.9052

# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10:                          fcs: cor_6h = 0.9531 rms_6h = 1.2388 nse_6h = 0.9032 r2s_6h = 0.9032
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10,stl1:                     fcs: cor_6h = 0.9526 rms_6h = 1.2442 nse_6h = 0.9022 r2s_6h = 0.9022
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10,stl1,r:                   fcs: cor_6h = 0.9491 rms_6h = 1.2903 nse_6h = 0.8950 r2s_6h = 0.8950
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10,stl1,r,msdrswrf:          fcs: cor_6h = 0.9463 rms_6h = 1.3241 nse_6h = 0.8889 r2s_6h = 0.8889
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10,stl1,r,msdrswrf,e:        fcs: cor_6h = 0.9387 rms_6h = 1.4114 nse_6h = 0.8724 r2s_6h = 0.8724
# - z,swvl2,msl,sd,swvl3,swvl1,tcc,stl2,tp,stl3,mslhf,v10,u10,stl1,r,msdrswrf,e,msshf:  fcs: cor_6h = 0.8951 rms_6h = 1.8277 nse_6h = 0.7662 r2s_6h = 0.7659




# Old Dropout experiment
# -                                                                                     CORR = 0.958 RMSE = 1.178 CORR-w = 0.970 RMSE-w = 0.786
# - tcc:                                                                                CORR = 0.957 RMSE = 1.193 CORR-w = 0.969 RMSE-w = 0.795
# - tcc,msl:                                                                            CORR = 0.958 RMSE = 1.170 CORR-w = 0.971 RMSE-w = 0.769
# - tcc,msl,tp:                                                                         CORR = 0.958 RMSE = 1.167 CORR-w = 0.971 RMSE-w = 0.774
# - tcc,msl,tp,swvl1:                                                                   CORR = 0.958 RMSE = 1.168 CORR-w = 0.970 RMSE-w = 0.784
# - tcc,msl,tp,swvl1,swvl2:                                                             CORR = 0.959 RMSE = 1.165 CORR-w = 0.971 RMSE-w = 0.770
# - tcc,msl,tp,swvl1,swvl2,u10:                                                         CORR = 0.957 RMSE = 1.183 CORR-w = 0.970 RMSE-w = 0.783
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3:                                                   CORR = 0.957 RMSE = 1.185 CORR-w = 0.970 RMSE-w = 0.781
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10:                                               CORR = 0.958 RMSE = 1.171 CORR-w = 0.972 RMSE-w = 0.762

# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r:                                             CORR = 0.955 RMSE = 1.214 CORR-w = 0.969 RMSE-w = 0.795
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z:                                           CORR = 0.954 RMSE = 1.225 CORR-w = 0.968 RMSE-w = 0.815
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd:                                        CORR = 0.954 RMSE = 1.227 CORR-w = 0.969 RMSE-w = 0.803
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf:                                  CORR = 0.954 RMSE = 1.228 CORR-w = 0.967 RMSE-w = 0.826
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e:                                CORR = 0.953 RMSE = 1.238 CORR-w = 0.964 RMSE-w = 0.852
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e,stl2:                           CORR = 0.952 RMSE = 1.248 CORR-w = 0.963 RMSE-w = 0.859
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e,stl2,stl3:                      CORR = 0.953 RMSE = 1.247 CORR-w = 0.965 RMSE-w = 0.849
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e,stl2,stl3,t2m:                  CORR = 0.951 RMSE = 1.267 CORR-w = 0.964 RMSE-w = 0.856
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e,stl2,stl3,t2m,msdrswrf:         CORR = 0.947 RMSE = 1.314 CORR-w = 0.963 RMSE-w = 0.865
# - tcc,msl,tp,swvl1,swvl2,u10,swvl3,v10,r,z,sd,mslhf,e,stl2,stl3,t2m,msdrswrf,stl1:    CORR = 0.906 RMSE = 1.726 CORR-w = 0.943 RMSE-w = 1.051



