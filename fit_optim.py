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
from sklearn.model_selection import train_test_split









'''
code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'


'''
code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'

era5_dir='/fmi/scratch/project_2002138/ERA-5_1p0deg/'




# Read own functions
sys.path.append(code_dir)
import functions as fcts




# Modeling approach. Random forest ='rfrs'; Gradient boosting = 'gbst'
appr = 'gbst' # 'rfrs' 'gbst'

# Whether to shuffle the additional in-sample samples
sffl = True  # False  True




idnt = appr+'_shuffle='+str(sffl)


#'rfrs_noShfl' # 'rfrs_noShfl' 'gbst_noShfl' 'rfrs_whfl' 'gbst_wShfl'



# Target data 
#obs = pd.read_csv(data_dir+'smeardata_20210224_set1.csv')
obs = pd.read_csv(data_dir+'smeardata_20220130_gapfilled.csv')
gap = pd.read_csv(data_dir+'smeardata_20220130_gapfill_method.csv')

obs['gapfill_method'] = gap['HYY_EDDY233.Qc_gapf_NEE']
obs['HYY_EDDY233.NEE'] = obs['HYY_EDDY233.NEE'].where(obs['gapfill_method'] == 0, other=np.nan)

obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

# Resampling to 6H, skip samples containing NaNs
#obs = obs[['HYY_EDDY233.F_c']].resample('6H').agg(pd.Series.mean, skipna=False)
obs = obs[['HYY_EDDY233.NEE']].resample('6H').agg(pd.Series.mean, skipna=False)
#obs = obs[['HYY_EDDY233.NEE']].resample('1H').agg(pd.Series.mean, skipna=False)

vrb = obs.columns[0]



















dropout = ['tcc', 'msl', 'tp', 'swvl1','swvl2','u10','swvl3','v10','r','z','sd','mslhf','e','stl2','stl3','t2m','msdrswrf','stl1'] 
dropout = [] 
#'tcc', 'msl', 'tp', 'swvl1','swvl2','u10','swvl3','v10','r','z','sd','mslhf','e','stl2','stl3','t2m','msdrswrf','stl1'

# Read ERA5 predictor data
era5_data = xr.open_dataset(data_dir+'era5_preprocessed.nc').sel(
    time=slice('1996-01-01', '2020-12-31'), 
    lat=slice(60,64),
    lon=slice(22,26)
    ).drop(dropout)


vrbs = list(era5_data.data_vars)
vrbs.sort()



'''
for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].plot(); plt.show()



for predictor in list(era5_data.data_vars.keys()):
    era5_data[predictor].mean(axis=(0)).plot(); plt.show()
'''






# Define model parameters
if appr == 'gbst':
    num_parallel_tree = 10 # 1 # 10
    max_depth = 7
    subsample=0.75
    colsample_bytree=0.75
    colsample_bynode=1
    early_stopping = True
    nstm = 500
    lrte = 0.075

if appr == 'rfrs':
    num_parallel_tree = 500
    max_depth = 14
    subsample=0.50
    colsample_bytree=1
    colsample_bynode=0.50
    early_stopping = False
    nstm = 1
    lrte = 1







# 1996–2018
# 1996–2020
all_yrs   = np.arange(1996,2021).astype(int)








models  = []
vld_yrs = []


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
                Z[vrb].loc[not_nans].values[:,np.newaxis], 'normal',100000)



Z[vrb+'_qt'].loc[not_nans] = transformed.squeeze()



# Additional in-sample sampling to study the sensitivity of the model on the size of fitting data
sample_sizes = np.arange(0.1,1.01,0.1)
sample_sizes = [1.0]


for ss in sample_sizes:
    Z['fcs_'+str(int(ss*100))]  = np.nan







# Prepare predictor matrix X 
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=all_yrs)
X.index = X.index.rename('time'); t_axis = X.index











# Make sure the same time steps are in X and in target Z and they are synchronized
X_time = X.index.values; Z_time = Z.index.values
X_in_Z = np.isin(Z_time, X_time); Z_in_X = np.isin(X_time, Z_time)
X = X.loc[Z_in_X]; Z = Z.loc[X_in_Z]






# Fit models in cross-validation framework
kf = KFold(5, shuffle=True); fold = 0
for trn_idx, tst_idx in kf.split(all_yrs):
    fold += 1
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
            learning_rate=lrte, 
            max_depth=max_depth,
            alpha=0.01,
            num_parallel_tree=num_parallel_tree,
            n_jobs=40, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,  
            colsample_bynode=colsample_bynode,  
            random_state=99)
        
        #sample_size = int(len(X.loc[t_common_fit])*ss)
        #t_fit_sampled = resample(t_common_fit, replace=False, n_samples=sample_size)
        if ss==1.0:
            smpl=0.999999999
        else:
            smpl=ss
        
        t_fit_sampled, _ = train_test_split(t_common_fit, train_size=smpl, shuffle=sffl)
        
        # Train
        verbose=True
        fitted_ensemble = fcts.fit_ensemble(
            X.loc[t_fit_sampled], Z[vrb+'_qt'].loc[t_fit_sampled], 
            X.loc[t_common_tst], Z[vrb+'_qt'].loc[t_common_tst], 
            base_estim, 'rmse', verbose=True, early_stopping=early_stopping)
        
        
        if ss==1.0:
            # Save the model if using full 100% sample size
            models.append(fitted_ensemble)
        
        
        print('Fitting',fold,ss,vrb,'took ' + fcts.stopwatch('stop', t))
        print(dropout)
        
        # Save the inverse transformed forecast
        prediction = fitted_ensemble.predict(X.iloc[tst_idx_x].values)[:,np.newaxis]
        Z['fcs_'+str(int(ss*100))].iloc[tst_idx_y] = qt_model.inverse_transform(prediction).squeeze() 
        



not_nans = ~ np.isnan(Z[vrb]).values

corr_6h = fcts.calc_corr(Z['fcs_100'].loc[not_nans], Z[vrb].loc[not_nans])
rmse_6h = fcts.calc_rmse(Z['fcs_100'].loc[not_nans], Z[vrb].loc[not_nans])

corr_1w = fcts.calc_corr(Z['fcs_100'].loc[not_nans].resample('1W').mean(), Z[vrb].loc[not_nans].resample('1W').mean())
rmse_1w = fcts.calc_rmse(Z['fcs_100'].loc[not_nans].resample('1W').mean(), Z[vrb].loc[not_nans].resample('1W').mean())

print(dropout)
print('CORR = '+str(corr_6h.round(3))+' RMSE = '+str(rmse_6h.round(3))+\
      ' CORR-w = '+str(corr_1w.round(3))+' RMSE-w = '+str(rmse_1w.round(3)))







# Save modeled results and models
Z.drop(columns=[vrb+'_qt']).to_csv(data_dir+'obs_mod_co2_'+idnt+'.csv')

for i,mdl in enumerate(models):
    mdl.save_model(data_dir+'model_'+str(i+1)+'_for_co2_'+idnt+'.json')





# Random forest hyperparameter optimization
# num_parallel_tree = 500	max_depth = 7	subsample=0.75	colsample_bytree=1	colsample_bynode=0.75			CORR=0.9309886416726332
# num_parallel_tree = 500	max_depth = 7	subsample=0.75	colsample_bytree=1	colsample_bynode=0.75			CORR=0.9332979935060194
# num_parallel_tree = 500	max_depth = 7	subsample=0.75	colsample_bytree=1	colsample_bynode=0.75			CORR=0.9331501211149369
# num_parallel_tree = 500	max_depth = 9	subsample=0.50	colsample_bytree=1	colsample_bynode=0.50			CORR=0.9373346857507269
# num_parallel_tree = 500	max_depth = 11	subsample=0.50	colsample_bytree=1	colsample_bynode=0.50			CORR=0.9385038499464846
# num_parallel_tree = 500	max_depth = 14	subsample=0.50	colsample_bytree=1	colsample_bynode=0.50			CORR=0.9386975750903893




# New Dropout experiment
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



# Old Dropout experiment
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




