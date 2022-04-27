#!/usr/bin/env python



# Read modules
import sys, glob, ast, importlib, datetime, itertools
import numpy as np
import pandas as pd
import xarray as xr




import xgboost as xgb


import matplotlib; #matplotlib.use('AGG') #matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import resample


import seaborn as sns










code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'


code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'

era5_dir='/fmi/scratch/project_2002138/ERA-5_1p0deg/'




# Read own functions
sys.path.append(code_dir)
import functions as fcts



all_yrs   = np.arange(1996,2019).astype(int)





'''
# Target data 
obs1 = pd.read_csv(data_dir+'smeardata_20210224_set1.csv')
obs1.index = pd.to_datetime(obs1[['Year','Month','Day','Hour','Minute','Second']])
obs1.index.name = 'time'

obs1 = obs1.loc['1996-01-01':'2018-12-31']

# Resampling to 6H, skip samples containing NaNs
obs1 = obs1[['HYY_EDDY233.F_c']].resample('6H').agg(pd.Series.mean, skipna=False)

vrb1 = obs1.columns[0]
'''
# Target data 
#obs = pd.read_csv(data_dir+'smeardata_20210224_set1.csv')
obs = pd.read_csv(data_dir+'smeardata_20220130_gapfilled.csv')
gap = pd.read_csv(data_dir+'smeardata_20220130_gapfill_method.csv')

obs['gapfill_method'] = gap['HYY_EDDY233.Qc_gapf_NEE']
obs['HYY_EDDY233.NEE'] = obs['HYY_EDDY233.NEE'].where(obs['gapfill_method'] == 0, other=np.nan)

obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

obs = obs.loc['1996-01-01':'2018-12-31']
# Resampling to 6H, skip samples containing NaNs
#obs = obs[['HYY_EDDY233.F_c']].resample('6H').agg(pd.Series.mean, skipna=False)
obs = obs[['HYY_EDDY233.NEE']].resample('6H').agg(pd.Series.mean, skipna=False)

vrb = obs.columns[0]




obs.plot(); plt.show()



obs_00 = obs.loc[obs.index.hour==0];    print((np.sum(np.isnan(obs_00))/obs_00.shape[0])*100)
obs_06 = obs.loc[obs.index.hour==6];    print((np.sum(np.isnan(obs_06))/obs_06.shape[0])*100)
obs_12 = obs.loc[obs.index.hour==12];   print((np.sum(np.isnan(obs_12))/obs_12.shape[0])*100)
obs_18 = obs.loc[obs.index.hour==18];   print((np.sum(np.isnan(obs_18))/obs_18.shape[0])*100)



n_datapoints = obs.shape[0];                print(n_datapoints)
n_missing = np.sum(np.isnan(obs.values));   print(n_missing)
n_not_missing = n_datapoints - n_missing;   print(n_not_missing)


# Plot climatology
obs_ann = pd.DataFrame(columns=range(1997, 2019))
for year in range(1997, 2019): 
    idx = obs.index.year == year
    len_year = len(obs.loc[idx].values[:1461])
    obs_ann[year] = np.squeeze(obs.loc[idx].values[:1460])
    print(year, len_year)


idx = obs.index.year == 1998
obs_ann.index = obs.loc[idx].index.values




clim_6h = obs_ann.mean(axis=1)
clim_1d = fcts.kz_filter(clim_6h, 4, 1) #fcts.kz_filter(obs_ann, 4, 1).mean(axis=1)
clim_1w = fcts.kz_filter(clim_6h, 28, 1) #fcts.kz_filter(obs_ann, 28, 1).mean(axis=1)
clim_1m = fcts.kz_filter(clim_6h, 112, 1) #fcts.kz_filter(obs_ann, 112, 1).mean(axis=1)


import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import matplotlib.dates as mdates



sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(12, 3))

clim_6h.plot(lw=0.5, c='k', label='6H mean', ax=ax, alpha=0.8)
clim_1w.plot(lw=2.5, c='k', label='1W mean', ax=ax, alpha=1)
ax.axhline(np.nanmean(obs),c='Orange', label='1Y mean')
ax.axhline(0,c='b', label='Neutral')


ax.set_ylabel('µmol m⁻² s⁻¹'); ax.set_xlabel('Time of year')
ax.set_title('Multi-year means of the measured net ecosystem CO2 exchange')
ax.legend(loc='lower left'); plt.tight_layout()
f.savefig(rslt_dir+'fig_climatology.pdf')
f.savefig(rslt_dir+'fig01.pdf')
f.savefig(rslt_dir+'fig_climatology.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')





















# Read ERA5 predictor data
era5_data = xr.open_dataset(data_dir+'era5_preprocessed.nc').sel(
    time=slice('1996-01-01', '2018-12-31'), 
    lat=slice(60,64),
    lon=slice(22,26)
    )


vrbs = list(era5_data.data_vars)
vrbs.sort()







# Prepare predictor matrix X 
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=np.arange(1996,2019))
X.index = X.index.rename('time'); t_axis = X.index


X.index = X.index.rename('time'); t_axis = X.index


# Shuffle X columns
import random; random.seed(99)
columns = list(X.columns)
random.shuffle(columns)

X = X[columns]

print(X)




















# Comparison of GB and RF
result_files = [
    'obs_mod_co2_gbst_shuffle=True.csv',
    'obs_mod_co2_gbst_shuffle=False.csv',
    'obs_mod_co2_rfrs_shuffle=True.csv',
    'obs_mod_co2_rfrs_shuffle=False.csv',
    ]


labels = {
    'gbst_shuffle=True':  'GB, random sampling',
    'gbst_shuffle=False': 'GB, non-random sampling',
    'rfrs_shuffle=True':  'RF, random sampling',
    'rfrs_shuffle=False': 'RF, non-random sampling'
    }


corr_subs = {}
rmse_subs = {}

percentages = np.arange(10,101,10)



data = obs.copy(deep=True)
data = data.rename(columns={vrb:    'Observed CO2 flux, 6-hourly'})

not_nans = ~ np.isnan(data['Observed CO2 flux, 6-hourly']).values

data['Observed CO2 flux, weekly']  = fcts.kz_filter(data['Observed CO2 flux, 6-hourly'].loc[not_nans], 28, 1)
data['Observed CO2 flux, weekly'].loc[~ not_nans] = np.nan



for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results and the model
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    #Z = pd.read_csv(data_dir+'obs_mod_co2_gbst_shuffle=True.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    #Z['fcs'] = Z['fcs_100'].copy(deep=True)
    
    models = []
    #mdl_files = np.sort(glob.glob(data_dir+'model_*_for_co2_gbst_shuffle=True.json'))
    mdl_files = np.sort(glob.glob(data_dir+'model_*_for_co2_'+idnt+'.json'))
    for mdl in mdl_files:
        model = xgb.XGBRegressor()
        model.load_model(mdl)
        models.append(model)
    
    
    
    
    
    # Data selection
    #data = Z.copy(deep=True)
    #data = data.rename(columns={'fcs':  labels[idnt]+', 6-hourly'})
    data[labels[idnt]+', 6-hourly'] = Z['fcs_100'].copy(deep=True)
    
    #not_nans_6h = ~ np.isnan(data['Observed CO2 flux, 6-hourly']).values
    #not_nans_wk = not_nans_6h # ndimage.binary_erosion(not_nans_6h, structure=np.ones(28)).astype(not_nans_6h.dtype)
    
    
    #data['Observed CO2 flux, weekly']  = fcts.kz_filter(data['Observed CO2 flux, 6-hourly'].loc[not_nans_6h], 28, 1)
    data[labels[idnt]+', weekly'] = fcts.kz_filter(data[labels[idnt]+', 6-hourly'].loc[not_nans], 28, 1)
    
    #data['Observed CO2 flux, weekly'].loc[~ not_nans] = np.nan
    
    
    corr = []
    rmse = []
    for pr in percentages:
        data['fcs'] = Z['fcs_'+str(pr)]
        o = data['Observed CO2 flux, 6-hourly'].values
        f = data['fcs'].values
        
        cor = fcts.calc_corr(o, f)
        rms = fcts.calc_rmse(o, f)
        
        corr.append(cor)
        rmse.append(rms)
    
    corr_subs[labels[idnt]] = corr
    rmse_subs[labels[idnt]] = rmse










sns.set_theme(style="white")
f, [ax1, ax2] = plt.subplots(1,2,figsize=(8,5))
#ax2 = ax1.twinx()

#plt.suptitle('Dependency of RMSE and CORR\non the amount of fitting data')
sns.set_palette("PuBuGn_d")
sns.set_palette("Blues", color_codes=True)



sns.lineplot(data=pd.DataFrame(corr_subs, index=percentages)[['GB, random sampling', 'GB, non-random sampling']], lw=1.5, palette="Blues_r", alpha=1, ax=ax1)
sns.lineplot(data=pd.DataFrame(corr_subs, index=percentages)[['RF, random sampling', 'RF, non-random sampling']], lw=1.5, palette="Reds_r", alpha=1, ax=ax1)
sns.lineplot(data=pd.DataFrame(rmse_subs, index=percentages)[['GB, random sampling', 'GB, non-random sampling']], lw=1.5, palette="Blues_r", alpha=1, ax=ax2)
sns.lineplot(data=pd.DataFrame(rmse_subs, index=percentages)[['RF, random sampling', 'RF, non-random sampling']], lw=1.5, palette="Reds_r", alpha=1, ax=ax2)
#sns.lineplot(data=pd.DataFrame(rmse_subs, index=percentages), lw=1.5, palette="Reds_r",  alpha=1, ax=ax2)


ax1.set_title('CORR')
ax2.set_title('RMSE')

ax1.set_xlabel('Amount [%]')
ax2.set_xlabel('Amount [%]')

plt.tight_layout(); ax1.legend(); ax2.legend()
f.savefig(rslt_dir+'fig_corr_rmse_percentages.pdf')
f.savefig(rslt_dir+'fig05.pdf')
f.savefig(rslt_dir+'fig_corr_rmse_percentages.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')










for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results and the model
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    Z['fcs'] = Z['fcs_100'].copy(deep=True)
    
    models = []
    mdl_files = np.sort(glob.glob(data_dir+'model_*_for_co2_'+idnt+'.json'))
    for mdl in mdl_files:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(mdl)
        models.append(xgb_model)
    
    
    
    # Predictor importance
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,len(models),figsize=(14, 8))
    for i,mdl in enumerate(models):
        out = mdl.predict(X.loc[X.index[0:2]].values)
        mdl.get_booster().feature_names = list(X.columns)
        print(len(mdl.get_booster().feature_names), len(X.columns))
        sns.set_theme(style="ticks")
        
        xgb.plot_importance(mdl, importance_type='gain', title='Gain,\nmodel '+str(i+1), ylabel='',
            max_num_features=40, show_values=False, height=0.7, ax=ax[i])
        #ax[i].set_xscale('log')
        ax[i].set_xlabel('')
    
    
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_gain_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_gain_'+idnt+'.png', dpi=200)
    if idnt=='gbst_shuffle=True':
        f.savefig(rslt_dir+'fig06.pdf')
    #plt.show()
    
    
    
    
    
    # Group the data embedded in the individual predictors for supplementary figs
    all_scores = pd.DataFrame(columns=['Model', 'Predictor variable', 'Lag', 'Grid cell', 'Mean gain'])
    row=0
    for i,mdl in enumerate(models):
        bst = mdl.get_booster()
        gains = np.array(list(bst.get_score(importance_type='gain').values()))
        features = np.array(list(bst.get_fscore().keys())) #np.array(X.columns)[sorted_idx]
        for ft, gn in zip(features, gains):
            print(i+1,ft,gn)
            try:
                _, prd, gc_lg = ft.split('_')
                lag = int(gc_lg[-2:])
                gcl = int(gc_lg[:-2])
            except:
                prd = ft
                lag = 0
                gcl = 4
            
            all_scores.loc[row] = (i+1, prd, lag, gcl, gn); row += 1 
    
    
    
    
    mean_scores = all_scores.groupby('Predictor variable').mean().sort_values('Mean gain')
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1,figsize=(8, 8))
    mean_scores.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('F score')
    ax.set_title('Mean gain over models, grid cells and lags')
    ax.set_xscale('log')
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_MEAN_gain_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_MEAN_gain_'+idnt+'.png', dpi=200)
    if idnt=='gbst_shuffle=True':
        f.savefig(rslt_dir+'figA01.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')
    
    
    
    mean_scores = all_scores.groupby('Grid cell').mean().sort_values('Mean gain')
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1,figsize=(8, 8))
    mean_scores.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('F score')
    ax.set_title('Mean gain over models, variables and lags')
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_GRIDCELL_gain_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_GRIDCELL_gain_'+idnt+'.png', dpi=200)
    if idnt=='gbst_shuffle=True':
        f.savefig(rslt_dir+'figA02.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')
    
    
    mean_scores = all_scores.groupby('Lag').mean().sort_values('Mean gain')
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1,figsize=(8, 8))
    mean_scores.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('F score')
    ax.set_title('Mean gain over models, variables, and grid cells')
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_LAG_gain_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_LAG_gain_'+idnt+'.png', dpi=200)
    if idnt=='gbst_shuffle=True':
        f.savefig(rslt_dir+'figA03.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')
    
    
    
    mean_scores = all_scores.groupby(['Grid cell', 'Lag']).mean().sort_values('Mean gain')
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1,figsize=(4, 12))
    mean_scores.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('F score')
    
    ax.set_title('Mean gain over models and variables, \ngrouped by (grid cells, lags)')
    plt.tight_layout()
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=6)
    f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_gain_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_gain_'+idnt+'.png', dpi=200)
    if idnt=='gbst_shuffle=True':
        f.savefig(rslt_dir+'figA04.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')











'''
# CALCULATING/PLOTTING THE CORRELATION MATRIX IS NOT FEASIBLE: TOO MANY PREDICTORS
sns.set_theme(style="white")

# Compute the correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.diverging_palette(250, 15, s=75, l=40,n=9, center='light', as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=.1, cbar_kws={"shrink": .7})

plt.tight_layout()

f.savefig(rslt_dir+'fig_feat_cross_correlations.pdf')
f.savefig(rslt_dir+'fig_feat_cross_correlations.png', dpi=200)

plt.show()




f, ax = plt.subplots(figsize=(10, 3))

to_plot = X[['Xv_t2m_4+1', 'Xv_t2m_2-1']].loc['2000-01-01':'2003-12-31']
corr = fcts.calc_corr(to_plot[to_plot.columns[0]], to_plot[to_plot.columns[1]]).round(3)

to_plot.plot(ax=ax); ax.set_title('Correlation: '+str(corr))
plt.tight_layout()

f.savefig(rslt_dir+'fig_collinearity_example.pdf')
f.savefig(rslt_dir+'fig_collinearity_example.png', dpi=200)

plt.show()
'''






# Time series, one sample year, OBS only
import locale
locale.setlocale(locale.LC_ALL, 'en_US')



sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(12, 3))
linewidths=[0.7, 3,]
colors=['k', 'k']

to_plot = data[['Observed CO2 flux, 6-hourly','Observed CO2 flux, weekly']]
for col, lw, clr in zip(to_plot.columns, linewidths, colors):
    yr_idx = data.index.year == 2013
    if col == 'Observed CO2 flux, 6-hourly': sel = yr_idx 
    if col == 'Observed CO2 flux, weekly':  sel = yr_idx #& not_nans
    to_plot[col].iloc[yr_idx].plot(lw=lw, c=clr, label=col, ax=ax, alpha=0.8)


ax.axhline(0,c='b'); ax.legend(loc='lower left'); plt.tight_layout(); 
f.savefig(rslt_dir+'fig_timeser_obs.pdf')
f.savefig(rslt_dir+'fig_timeser_obs.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')





# Time series, three sample years
sns.set_theme(style="whitegrid")

smpl_yrs = [2016, 1999, 2004] #
#smpl_yrs = random.sample(list(all_yrs), 3)

f, ax = plt.subplots(len(smpl_yrs), 1, figsize=(12, 3*len(smpl_yrs)))
linewidths=[0.7, 0.7, 5, 5, 3]
colors=['red', 'k', 'blue', 'red', 'k']
cols = ['GB, random sampling, 6-hourly', 'Observed CO2 flux, 6-hourly',
        'RF, random sampling, weekly','GB, random sampling, weekly', 'Observed CO2 flux, weekly']

for i,yr in enumerate(smpl_yrs):
    for col, lw, clr in zip(cols, linewidths, colors):
        yr_idx = data.index.year == yr
        if col == 'Observed CO2 flux, 6-hourly': sel = yr_idx
        if col == 'Observed CO2 flux, weekly':   sel = yr_idx 
        else: sel = np.ones(data.index.shape, bool)
        
        to_plot = data[col].copy(deep=True)
        to_plot.iloc[~sel] = np.nan
        to_plot.loc[str(yr)+'-01-01':str(yr)+'-12-31'].plot(lw=lw, c=clr, label=col, ax=ax[i], alpha=0.8)
        ax[i].set_xlim(str(yr)+'-01-01', str(yr)+'-12-31')
        ax[i].axhline(0,c='b'); 
        ax[i].set_ylabel('µmol m⁻² s⁻¹'); #ax[i].set_xlabel('Time of year')


ax[0].legend(loc='lower left')
plt.tight_layout(); 
f.savefig(rslt_dir+'fig_timeser2.pdf')
f.savefig(rslt_dir+'fig_timeser2.png', dpi=200)
#plt.show(); 
plt.clf(); plt.close('all')









# Diurnal and monthly decomposition and analysis of CORR and RMSE skill
mndi_corr_grb = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
mndi_rmse_grb = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
mndi_corr_rfr = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
mndi_rmse_rfr = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
for hr in (0, 6, 12, 18):
    for mn in np.arange(1,13):
        idx = (data.index.hour == hr) & (data.index.month == mn)
        obs = data['Observed CO2 flux, 6-hourly'  ][idx & not_nans]
        grb = data['GB, random sampling, 6-hourly'][idx & not_nans]
        rfr = data['RF, random sampling, 6-hourly'][idx & not_nans]
        cor_grb, _ = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        rms_grb, _ = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        cor_rfr, _ = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        rms_rfr, _ = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        print(mn, hr, cor_grb, cor_rfr, rms_grb, rms_rfr)
        mndi_corr_grb.loc[mn,hr] = cor_grb[2]
        mndi_rmse_grb.loc[mn,hr] = rms_grb[2]
        mndi_corr_rfr.loc[mn,hr] = cor_rfr[2]
        mndi_rmse_rfr.loc[mn,hr] = rms_rfr[2]


mndi_corr_grb = mndi_corr_grb.astype(float)
mndi_rmse_grb = mndi_rmse_grb.astype(float)
mndi_corr_rfr = mndi_corr_rfr.astype(float)
mndi_rmse_rfr = mndi_rmse_rfr.astype(float)





# Evolution of CORR and RMSE skill over time
boot_rms_grb = []
boot_cor_grb = []
boot_rms_rfr = []
boot_cor_rfr = []
cmpt = []

#to_plot = data[['Observed CO2 flux, 6-hourly','Observed CO2 flux, weekly']]
for yr in all_yrs:
    yr_idx = data.index.year == yr
    completeness = (yr_idx & not_nans).sum() / (365.24*4) * 100
    obs = data['Observed CO2 flux, 6-hourly'  ][yr_idx & not_nans]
    grb = data['GB, random sampling, 6-hourly'][yr_idx & not_nans]
    rfr = data['RF, random sampling, 6-hourly'][yr_idx & not_nans]
    _, bts_rms_grb = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    _, bts_cor_grb = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    _, bts_rms_rfr = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    _, bts_cor_rfr = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    boot_rms_grb.append(bts_rms_grb)
    boot_cor_grb.append(bts_cor_grb)
    boot_rms_rfr.append(bts_rms_rfr)
    boot_cor_rfr.append(bts_cor_rfr)
    cmpt.append(completeness.round(1))
    print(yr, completeness.round(1))





from matplotlib.gridspec import GridSpec
import matplotlib as mpl

fig=plt.figure(figsize=(16,8))

gs=GridSpec(2,3) # 2 rows, 3 columns

ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[1,0]) # Second row, first column
ax3=fig.add_subplot(gs[0,1:]) # First row, second column
ax4=fig.add_subplot(gs[1,1:]) # First row, third column


sns.set_theme(style="white")
sns.heatmap(mndi_rmse_grb, cmap='hot_r', annot=True, ax=ax1)
sns.heatmap(mndi_corr_grb, cmap='YlGnBu_r', annot=True, ax=ax2)
ax1.set_title('RMSE'); ax1.set_xlabel('Hour');  ax1.set_ylabel('Month'); 
ax2.set_title('CORR'); ax2.set_xlabel('Hour');  ax2.set_ylabel('Month'); 


sns.set_theme(style="whitegrid")


sel = np.array(cmpt) > 30 
nan = ~sel


to_plot1 = pd.DataFrame(np.array(boot_rms_rfr).T, columns=all_yrs); to_plot1[to_plot1.columns[nan]] = np.nan
to_plot1 = pd.melt(to_plot1, var_name='Year', value_name='RMSE')
to_plot1['Model'] = 'RF'
to_plot2 = pd.DataFrame(np.array(boot_rms_grb).T, columns=all_yrs); to_plot2[to_plot2.columns[nan]] = np.nan
to_plot2 = pd.melt(to_plot2, var_name='Year', value_name='RMSE')
to_plot2['Model'] = 'GB'
to_plot = pd.concat([to_plot1, to_plot2], ignore_index=True, sort=False)

g = sns.boxenplot(data=to_plot, x='Year', y='RMSE', hue='Model', ax=ax3, width=0.7, color='b', k_depth=10, showfliers=False, linewidth=1 ) 


print(to_plot.groupby('Year').median())


#ax3.set_title('RMSE'); 
ax3.legend(loc='upper left'); ax3.set_xlabel('')
ax3.set_xticks(ax3.get_xticks()+.5, minor=True)
ax3.yaxis.grid(False); ax3.xaxis.grid(True, which='minor') 

labels = g.get_xticklabels(); g.set_xticklabels(labels, rotation=30)

to_plot1 = pd.DataFrame(np.array(boot_cor_rfr).T, columns=all_yrs); to_plot1[to_plot1.columns[nan]] = np.nan
to_plot1 = pd.melt(to_plot1, var_name='Year', value_name='CORR')
to_plot1['Model'] = 'RF'
to_plot2 = pd.DataFrame(np.array(boot_cor_grb).T, columns=all_yrs); to_plot2[to_plot2.columns[nan]] = np.nan
to_plot2 = pd.melt(to_plot2, var_name='Year', value_name='CORR')
to_plot2['Model'] = 'GB'
to_plot = pd.concat([to_plot1, to_plot2], ignore_index=True, sort=False)

h = sns.boxenplot(data=to_plot, x='Year', y='CORR', hue='Model', ax=ax4, width=0.7, color='r', k_depth=10, showfliers=False, linewidth=1 ) 

print(to_plot.groupby('Year').median())





#ax4.set_title('CORR'); 
ax4.legend(loc='upper left'); ax4.set_xlabel('')
ax4.set_xticks(ax4.get_xticks()+.5, minor=True)
ax4.yaxis.grid(False); ax4.xaxis.grid(True, which='minor') 

labels = h.get_xticklabels(); h.set_xticklabels(labels, rotation=30)


plt.figtext(0.02,0.94,'a)',fontsize=25,fontstyle='italic',fontweight='bold')
plt.figtext(0.34,0.94,'b)',fontsize=25,fontstyle='italic',fontweight='bold')


#plt.suptitle('Bootstrap estimates of RMSE and Pearson correlation with 10$^4$ samples, estimated from 6-hourly data')

fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90,wspace=0.2,hspace=0.3)
fig.savefig(rslt_dir+'fig_RMSE_CORR_time_GB_vs_RF.pdf',)# bbox_inches='tight')
fig.savefig(rslt_dir+'fig04.pdf',)# bbox_inches='tight')
fig.savefig(rslt_dir+'fig_RMSE_CORR_time_GB_vs_RF.png', dpi=200,)# bbox_inches='tight')
#plt.show(); 
plt.clf(); plt.close('all')











# Scatter plots / density histograms
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, SymLogNorm
sns.set_theme(style="ticks")
f, axes = plt.subplots(2,2, figsize=(10,10))#, sharex=True, sharey=True)




to_plot = [ data.iloc[not_nans][['Observed CO2 flux, 6-hourly', 'GB, random sampling, 6-hourly']],
            data.iloc[not_nans][['Observed CO2 flux, 6-hourly', 'RF, random sampling, 6-hourly']], 
            data.iloc[not_nans][['Observed CO2 flux, weekly',   'GB, random sampling, weekly']],
            data.iloc[not_nans][['Observed CO2 flux, weekly',   'RF, random sampling, weekly']] ]


for ax, pl in zip(axes.ravel(), to_plot):
    
    corr = fcts.calc_corr(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    rmse = fcts.calc_rmse(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    
    # Bootstrap estimates
    rms_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    cor_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
    print('RMSE bootstrap for',pl.columns[1],':',rms_b)
    print('CORR bootstrap for',pl.columns[1],':',cor_b)
    
    vmin, vmax = np.ravel(pl).min()-0.5, np.ravel(pl).max()+0.5
    bin_edges = np.linspace(vmin, vmax, 50)
    ax.hist2d(pl[pl.columns[0]], pl[pl.columns[1]], bins=bin_edges, cmap='YlGnBu_r', norm=LogNorm(1))
    
    ax.plot([vmin, vmax], [vmin, vmax], color='b', linestyle="--")
    ax.axhline(0, color='b', linestyle="--")
    ax.axvline(0, color='b', linestyle="--")
    
    ax.set_xlim([vmin, vmax]); ax.set_ylim([vmin, vmax])
    ax.set_xlabel(pl.columns[0]); ax.set_ylabel(pl.columns[1])
    
    ax.text(0.1,0.85,'CORR: '+str(corr)+'\nRMSE: '+str(rmse), transform=ax.transAxes)
    



'''
axes.ravel()[0].set_title('6-hourly CO2 flux [µmol m⁻² s⁻¹]')
axes.ravel()[1].set_title('6-hourly CO2 flux [µmol m⁻² s⁻¹]')
axes.ravel()[2].set_title('Weekly CO2 flux [µmol m⁻² s⁻¹]')
axes.ravel()[3].set_title('Weekly CO2 flux [µmol m⁻² s⁻¹]')
'''

plt.tight_layout(); 
f.savefig(rslt_dir+'fig_scatter.pdf')
f.savefig(rslt_dir+'fig03.pdf')
f.savefig(rslt_dir+'fig_scatter.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')




