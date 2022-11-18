#!/usr/bin/env python



# Read modules
import sys, glob, ast, importlib, datetime, itertools
import numpy as np
import pandas as pd
import xarray as xr

import shap


import xgboost as xgb


import matplotlib; #matplotlib.use('AGG') #matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
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


n_repeats = 8



# Target data 
obs = pd.read_csv(data_dir+'smeardata_20220130_gapfilled.csv')
gap = pd.read_csv(data_dir+'smeardata_20220130_gapfill_method.csv')

obs['gapfill_method'] = gap['HYY_EDDY233.Qc_gapf_NEE']
obs['HYY_EDDY233.NEE'] = obs['HYY_EDDY233.NEE'].where(obs['gapfill_method'] == 0, other=np.nan)

obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

obs = obs.loc['1996-01-01':'2018-12-31']

# Resampling to 6H, skip samples containing NaNs
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
ax.axhline(np.nanmean(obs), c='Orange', label='1Y mean')
ax.axhline(0,c='b', label='Neutral')


ax.set_ylabel('µmol m⁻² s⁻¹'); ax.set_xlabel('Time of year')
ax.set_title('Multi-year means of the measured net ecosystem CO$_2$ exchange')
ax.legend(loc='lower left'); 
#fcts.format_axes(ax)
plt.tight_layout()
f.savefig(rslt_dir+'fig_climatology.pdf')
f.savefig(rslt_dir+'fig01_.pdf')
f.savefig(rslt_dir+'fig_climatology.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')





















# Read ERA5 predictor data
dropout = ['z','swvl2','msl','sd','swvl3','swvl1','tcc','stl2','tp'] 
dDeg = 0.6; lat_h=61.85; lon_h=24.283
era5_data = xr.open_dataset(data_dir+'era5_preprocessed_0p25deg.nc').sel(
    time=slice('1996-01-01', '2020-12-31'), 
    lat=slice(lat_h - dDeg, lat_h + dDeg), #lat=slice(60,64),
    lon=slice(lon_h - dDeg, lon_h + dDeg), #lon=slice(22,26)
    ).drop(dropout)


vrbs = list(era5_data.data_vars)
vrbs.sort()







# Prepare predictor matrix X 
X    = fcts.combine_and_define_lags(era5_data, np.arange(-2,3), all_yrs=np.arange(1996,2019))
X.index = X.index.rename('time'); t_axis = X.index

# Predictor data for a control experiment WITHOUT spatial or temporal lags
C    = fcts.combine_and_define_lags(era5_data.sel(lat=61.85, lon=24.283, method='nearest'), [0], all_yrs=np.arange(1996,2019))
C.index = C.index.rename('time')





print(X)
print(C)











# Optimization results from runs
for approach in ['GB', 'RF']:
    
    for sfl in ['True','False']:
        list_of_optim_results = []
        opt_params_files = glob.glob(data_dir+'best_params_Bayes_optim_*_'+approach+'*'+sfl+'.csv')
        for file in opt_params_files:
            data = pd.read_csv(file, usecols=['param','value'], index_col='param')
            list_of_optim_results.append(data)

        opt_params_mdn  = dict(zip(data.index, np.median(list_of_optim_results, axis=0).squeeze())) 
        opt_params_std  = dict(zip(data.index, np.std(list_of_optim_results, axis=0).squeeze())) 
        opt_params_min  = dict(zip(data.index, np.min(list_of_optim_results, axis=0).squeeze())) 
        opt_params_max  = dict(zip(data.index, np.max(list_of_optim_results, axis=0).squeeze())) 


        opt_params_mdn['max_depth'] = int(opt_params_mdn['max_depth'])

        if approach=='GB':
            opt_params_mdn['n_estimators'] = int(opt_params_mdn['n_estimators'])
            opt_params_mdn['num_parallel_tree'] = 1
        if approach=='RF':
            opt_params_mdn['n_estimators'] = 1
            opt_params_mdn['num_parallel_tree'] = int(opt_params_mdn['num_parallel_tree'])



        print('Optimized parameters for '+ approach+' shuffle='+sfl+', mdn:\n',opt_params_mdn)
        #print('Optimized parameters for '+ approach+' shuffle='+sfl+', std:\n',opt_params_std)
        print('Optimized parameters for '+ approach+' shuffle='+sfl+', min:\n',opt_params_min)
        print('Optimized parameters for '+ approach+' shuffle='+sfl+', max:\n',opt_params_max)
        print('\n')










# Comparison of GB and RF
result_files = [
    'obs_mod_co2_GB_shuffle=True.csv',
    'obs_mod_co2_GB_shuffle=False.csv',
    'obs_mod_co2_RF_shuffle=True.csv',
    'obs_mod_co2_RF_shuffle=False.csv',
    ]


labels = {
    'GB_shuffle=True':  'GB, random sampling',
    'GB_shuffle=False': 'GB, non-random sampling',
    'RF_shuffle=True':  'RF, random sampling',
    'RF_shuffle=False': 'RF, non-random sampling'
    }


corr_subs = {}
rmse_subs = {}
r2ss_subs = {}


percentages = np.arange(10,101,10)



data = obs.copy(deep=True)
data = data.rename(columns={vrb: 'Observed CO2 flux, 6-hourly'})

not_nans = ~ np.isnan(data['Observed CO2 flux, 6-hourly']).values

data['Observed CO2 flux, weekly']  = fcts.kz_filter(data['Observed CO2 flux, 6-hourly'].loc[not_nans], 28, 1)
data['Observed CO2 flux, weekly'].loc[~ not_nans] = np.nan





for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results and the model
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    
    Z_reorg = pd.DataFrame()
    for pr in percentages.astype(str):
        
        fcs_cols = []; ctr_cols = []
        for repeat in np.arange(n_repeats).astype(str):
            fcs_cols.append('fcs_'+repeat+'_'+pr)
            ctr_cols.append('ctr_'+repeat+'_'+pr)
        
        to_melt = Z[fcs_cols].rename(columns=dict(zip(fcs_cols, range(n_repeats))))
        melted_fcs = pd.melt(to_melt,var_name='repeat', value_name='fcs_'+pr, ignore_index=False)
        to_melt = Z[ctr_cols].rename(columns=dict(zip(ctr_cols, range(n_repeats))))
        melted_ctr = pd.melt(to_melt,var_name='repeat', value_name='ctr_'+pr, ignore_index=False)
        
        Z_reorg['repeat']  = melted_fcs['repeat']
        Z_reorg['time']    = melted_fcs.index
        Z_reorg['fcs_'+pr] = melted_fcs['fcs_'+pr]
        Z_reorg['ctr_'+pr] = melted_ctr['ctr_'+pr]
        
        Z['fcs_'+pr] = Z_reorg.groupby('time').mean()['fcs_'+pr].copy(deep=True)
        Z['ctr_'+pr] = Z_reorg.groupby('time').mean()['ctr_'+pr].copy(deep=True)
    
    
    '''
    models = []
    mdl_files = np.sort(glob.glob(data_dir+'model_*_'+idnt+'*.json'))
    for mdl in mdl_files:
        model = xgb.XGBRegressor()
        model.load_model(mdl)
        models.append(model)
    '''
    
    
    
    
    # Data selection
    #data = Z.copy(deep=True)
    #data = data.rename(columns={'fcs':  labels[idnt]+', 6-hourly'})
    data[labels[idnt]+', 6-hourly'] = Z_reorg.groupby('time').mean()['fcs_100'].copy(deep=True)
    data[labels[idnt]+', 6-hourly, CTRL'] = Z_reorg.groupby('time').mean()['ctr_100'].copy(deep=True)
    
    #not_nans_6h = ~ np.isnan(data['Observed CO2 flux, 6-hourly']).values
    #not_nans_wk = not_nans_6h # ndimage.binary_erosion(not_nans_6h, structure=np.ones(28)).astype(not_nans_6h.dtype)
    
    
    #data['Observed CO2 flux, weekly']  = fcts.kz_filter(data['Observed CO2 flux, 6-hourly'].loc[not_nans_6h], 28, 1)
    data[labels[idnt]+', weekly'] = fcts.kz_filter(data[labels[idnt]+', 6-hourly'].loc[not_nans], 28, 1)
    data[labels[idnt]+', weekly, CTRL'] = fcts.kz_filter(data[labels[idnt]+', 6-hourly, CTRL'].loc[not_nans], 28, 1)
    
    #data['Observed CO2 flux, weekly'].loc[~ not_nans] = np.nan
    
    
    corr = []; corc = []
    rmse = []; rmsc = []
    r2ss = []; r2sc = []
    for pr in percentages.astype(str):
        
        data['fcs'] = Z['fcs_'+pr].copy(deep=True)
        data['ctr'] = Z['ctr_'+pr].copy(deep=True)
        o = data['Observed CO2 flux, 6-hourly'].values
        f = data['fcs'].values
        c = data['ctr'].values
        
        
        corr.append(fcts.calc_corr(o, f))
        rmse.append(fcts.calc_rmse(o, f))
        r2ss.append(fcts.calc_r2ss(o, f))
        corc.append(fcts.calc_corr(o, c))
        rmsc.append(fcts.calc_rmse(o, c))
        r2sc.append(fcts.calc_r2ss(o, c))
    
    corr_subs[labels[idnt]] = corr
    rmse_subs[labels[idnt]] = rmse
    r2ss_subs[labels[idnt]] = r2ss
    corr_subs[labels[idnt]+', CTRL'] = corc
    rmse_subs[labels[idnt]+', CTRL'] = rmsc
    r2ss_subs[labels[idnt]+', CTRL'] = r2sc










sns.set_theme(style="whitegrid")
f, axes = plt.subplots(1,3,figsize=(12,4), sharey=False)
#ax2 = ax1.twinx()

#plt.suptitle('Dependency of RMSE and CORR\non the amount of fitting data')
#sns.set_palette("PuBuGn_d")
#sns.set_palette("Blues", color_codes=True)


sns.lineplot(data=pd.DataFrame(r2ss_subs, index=percentages)[['GB, random sampling', 'GB, non-random sampling', 
    ]], lw=1.5, palette="Blues_r", alpha=1, ax=axes[0], legend=False)
sns.lineplot(data=pd.DataFrame(r2ss_subs, index=percentages)[['RF, random sampling', 'RF, non-random sampling', 
    'RF, random sampling, CTRL']], lw=1.5, palette="Reds_r", alpha=1, ax=axes[0], legend=False)
sns.lineplot(data=pd.DataFrame(corr_subs, index=percentages)[['GB, random sampling', 'GB, non-random sampling', 
    ]], lw=1.5, palette="Blues_r", alpha=1, ax=axes[1])
sns.lineplot(data=pd.DataFrame(corr_subs, index=percentages)[['RF, random sampling', 'RF, non-random sampling', 
    'RF, random sampling, CTRL']], lw=1.5, palette="Reds_r", alpha=1, ax=axes[1])
sns.lineplot(data=pd.DataFrame(rmse_subs, index=percentages)[['GB, random sampling', 'GB, non-random sampling', 
    ]], lw=1.5, palette="Blues_r", alpha=1, ax=axes[2], legend=False)
sns.lineplot(data=pd.DataFrame(rmse_subs, index=percentages)[['RF, random sampling', 'RF, non-random sampling', 
    'RF, random sampling, CTRL']], lw=1.5, palette="Reds_r", alpha=1, ax=axes[2], legend=False)


axes[0].set_title('R2SC'); axes[1].set_title('CORR'); axes[2].set_title('RMSE')
axes[0].set_xlabel('Amount [%]'); axes[1].set_xlabel('Amount [%]') ; axes[2].set_xlabel('Amount [%]')
#axes[0].set_ylim([0.78,0.92]); axes[1].set_ylim([0.88,0.96]); axes[2].set_ylim([1.2,2.0])
axes[1].legend(fontsize='x-small')

plt.tight_layout(); 
f.savefig(rslt_dir+'fig_r2ss_corr_rmse_percentages.pdf')
f.savefig(rslt_dir+'fig05.pdf')
f.savefig(rslt_dir+'fig_r2ss_rmse_percentages.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')










# SHAP value analysis
# Warning! This analysis is slow to compute
for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results 
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    
    Z_reorg = pd.DataFrame()
    for pr in percentages.astype(str):
        
        fcs_cols = []; ctr_cols = []
        for repeat in np.arange(n_repeats).astype(str):
            fcs_cols.append('fcs_'+repeat+'_'+pr)
            ctr_cols.append('ctr_'+repeat+'_'+pr)
        
        to_melt = Z[fcs_cols].rename(columns=dict(zip(fcs_cols, range(n_repeats))))
        melted_fcs = pd.melt(to_melt,var_name='repeat', value_name='fcs_'+pr, ignore_index=False)
        to_melt = Z[ctr_cols].rename(columns=dict(zip(ctr_cols, range(n_repeats))))
        melted_ctr = pd.melt(to_melt,var_name='repeat', value_name='ctr_'+pr, ignore_index=False)
        
        Z_reorg['repeat']  = melted_fcs['repeat']
        Z_reorg['time']    = melted_fcs.index
        Z_reorg['fcs_'+pr] = melted_fcs['fcs_'+pr]
        Z_reorg['ctr_'+pr] = melted_ctr['ctr_'+pr]
        
        Z['fcs_'+pr] = Z_reorg.groupby('time').mean()['fcs_'+pr].copy(deep=True)
        Z['ctr_'+pr] = Z_reorg.groupby('time').mean()['ctr_'+pr].copy(deep=True)
    
    
    # Read the models
    models = []
    mdl_files = np.sort(glob.glob(data_dir+'model_*_*_'+idnt+'.json'))
    for mdl in mdl_files:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(mdl)
        models.append(xgb_model)
    
        
    '''
    # SHAP Predictor importance
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,len(models),figsize=(45, 8))
    #kf = KFold(5, shuffle=True, random_state=99);
    kf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=99) 
    fold = 0
    split_count=0
    i = 0
    #for i,mdl in enumerate(models):
    for trn_idx, tst_idx in kf.split(all_yrs):
        mdl = models[i]
        fold = np.mod(split_count,5) + 1
        rept = np.floor_divide(split_count,5) 
        
        sns.set_theme(style="ticks")
        
        sorted_names, sorted_SHAP = fcts.extract_shap_values(X.iloc[tst_idx], models[i])
        
        ax[i].barh(sorted_names[0:40], sorted_SHAP[0:40])#, ax=ax[i])
        #ax[i].set_xscale('log')
        ax[i].set_xlabel('')
        i += 1
        split_count += 1
    
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_shap_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_shap_'+idnt+'.png', dpi=200)
    if s:
        f.savefig(rslt_dir+'fig06.pdf')
    
    ''' 
        
    
    
    
    # Group the data embedded in the individual predictors for supplementary figs
    all_scores = pd.DataFrame(columns=['Model', 'Predictor variable', 'Lag', 'Grid cell', 'Mean SHAP'])
    
    #for i,mdl in enumerate(models):
    #kf = KFold(5, shuffle=True, random_state=99);
    kf = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=99) 
    row=0
    fold = 0
    split_count=0
    i = 0
    for trn_idx, tst_idx in kf.split(all_yrs):
        
        sorted_names, sorted_SHAP = fcts.extract_shap_values(X.iloc[tst_idx], models[i])
        
        #for ft, gn in zip(features, gains):
        for ft, sh in zip(sorted_names, sorted_SHAP):
            print(i+1,ft,sh)
            try:
                _, prd, gc_lg = ft.split('_')
                lag = int(gc_lg[-2:])
                gcl = int(gc_lg[:-2])
            except:
                prd = ft
                lag = 0
                gcl = 4
            
            all_scores.loc[row] = (i+1, prd, lag, gcl, sh); row += 1 
        
        i += 1
        split_count += 1
    
    
    
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,3,figsize=(12, 8))
    
    mean_scores = all_scores.groupby('Predictor variable').mean()['Mean SHAP'].sort_values()
    mean_scores.plot.barh(ax=ax[0], width=0.3, legend=False)
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax[0].set_xlabel('SHAP value')
    ax[0].set_title('mean(|SHAP|) over\nmodels, grid cells and lags')
    
    mean_scores = all_scores.groupby('Grid cell').mean()['Mean SHAP'].sort_values()
    mean_scores.plot.barh(ax=ax[1], width=0.75, legend=False)
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax[1].set_xlabel('SHAP value')
    ax[1].set_title('mean(|SHAP|) over\nmodels, variables and lags')
    
    mean_scores = all_scores.groupby('Lag').mean()['Mean SHAP'].sort_values()
    mean_scores.plot.barh(ax=ax[2], width=0.15, legend=False)
    ax[2].xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax[2].set_xlabel('SHAP value')
    ax[2].set_title('mean(|SHAP|) over\nmodels, variables, and grid cells')
    
    plt.figtext(0.06,0.96,'(a)',fontsize=25,fontstyle='italic',fontweight='bold')
    plt.figtext(0.37,0.96,'(b)',fontsize=25,fontstyle='italic',fontweight='bold')
    plt.figtext(0.69,0.96,'(c)',fontsize=25,fontstyle='italic',fontweight='bold')
    
    plt.tight_layout(); 
    f.savefig(rslt_dir+'fig_feat_importance_all_SHAP_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_all_SHAP_'+idnt+'.png', dpi=200)
    if idnt=='GB_shuffle=True':
        f.savefig(rslt_dir+'fig06.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')
    
    
    
    mean_scores = all_scores.groupby(['Grid cell', 'Lag']).mean()['Mean SHAP'].sort_values()
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1,figsize=(4, 12))
    mean_scores.plot.barh(ax=ax, legend=False)
    ax.set_xlabel('SHAP value')
    
    ax.set_title('mean(|SHAP|) over models and variables, \ngrouped by (grid cells, lags)')
    plt.tight_layout(); plt.locator_params(axis='x', nbins=4)
    #ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=6)
    f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_SHAP_'+idnt+'.pdf')
    f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_SHAP_'+idnt+'.png', dpi=200)
    if idnt=='GB_shuffle=True':
        f.savefig(rslt_dir+'figA04.pdf')
    
    #plt.show()
    plt.clf(); plt.close('all')
    
    
    
    mean_scores = all_scores.groupby(['Grid cell']).mean()['Mean SHAP'].sort_values()
    gcells = xr.full_like(era5_data['t2m'][0].drop('time').stack(gridcell=('lat','lon')), np.nan).rename('grid_cells')
    gcells.attrs = {'units': 'Number'}
    gcells = gcells.to_dataset()
    gcells['SHAP'] = xr.full_like(gcells['gridcell'], np.nan).astype(float)
    gcells['Number'] = xr.full_like(gcells['gridcell'], np.nan)
    for i,gc in enumerate(gcells.gridcell.values): 
        print(i+1,gc)
        gcells['SHAP'].loc[{'gridcell':gc}] = float(mean_scores.loc[i+1])
        gcells['Number'].loc[{'gridcell':gc}] = i+1
    
    f, ax = plt.subplots(1,1,figsize=(8, 8))
    lats,lons = gcells.unstack('gridcell').lat.values, gcells.unstack('gridcell').lon.values
    data_shap = pd.DataFrame(gcells['SHAP'].unstack('gridcell'), index=lats, columns=lons).sort_index(ascending=False)
    data_nmbr = pd.DataFrame(gcells['Number'].unstack('gridcell'), index=lats, columns=lons).sort_index(ascending=False)
    h = sns.heatmap(data_shap, annot=data_nmbr, cmap="YlGnBu_r",linewidths=.3,zorder=0,ax=ax); # plt.show()
    #s = sns.scatterplot(y=[61.85],x=[24.283],s=200,color='r',zorder=99,ax=ax) 
    ax.scatter(y=[61.85],x=[24.283],s=200,color='r',zorder=99) 
    plt.tight_layout()
    f.savefig(rslt_dir+'fig_feat_importance_GRIDCELL_spatial_SHAP_'+idnt+'.pdf')
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



















# Diurnal and monthly decomposition and analysis of R2SC, CORR, and RMSE skill
mndi_corr = {}
mndi_rmse = {}
mndi_r2ss = {}


for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results 
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    
    not_nans_Z = ~ np.isnan(Z['HYY_EDDY233.NEE']).values
    corr_out = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
    rmse_out = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
    r2ss_out = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
    
    for mn in np.arange(1,13):
        for hr in (0, 6, 12, 18):
            idx = (Z.index.hour == hr) & (Z.index.month == mn)
            
            
            obs = Z['HYY_EDDY233.NEE'][idx & not_nans_Z]
            cor = []; r2s = []; rms = []
            for repeat in np.arange(n_repeats).astype(str):
                cor.append(fcts.calc_corr(obs, Z['fcs_'+repeat+'_100'][idx & not_nans_Z]))
                r2s.append(fcts.calc_r2ss(obs, Z['fcs_'+repeat+'_100'][idx & not_nans_Z]))
                rms.append(fcts.calc_rmse(obs, Z['fcs_'+repeat+'_100'][idx & not_nans_Z]))
            
            cor = np.percentile(cor, [0.5, 2.5, 50, 97.5, 99.5])
            r2s = np.percentile(r2s, [0.5, 2.5, 50, 97.5, 99.5])
            rms = np.percentile(rms, [0.5, 2.5, 50, 97.5, 99.5])
            
            print(mn, hr, r2s, rms, cor)
            
            corr_out.loc[mn,hr] = cor[2]
            rmse_out.loc[mn,hr] = rms[2]
            r2ss_out.loc[mn,hr] = r2s[2]
            
    
    mndi_corr[idnt] = corr_out.astype(float)
    mndi_rmse[idnt] = rmse_out.astype(float)
    mndi_r2ss[idnt] = r2ss_out.astype(float)
    
    









# Evolution of CORR, R2 and RMSE skill over time
yrly_corr = {}
yrly_rmse = {}
yrly_r2ss = {}
cmpt = {}


for fle in result_files:
    
    idnt = fle[12:-4]
    print(idnt)
    
    # Read modeled results and the model
    Z = pd.read_csv(data_dir+'obs_mod_co2_'+idnt+'.csv', index_col=0)
    Z.index = pd.to_datetime(Z.index)
    
    yrly_corr[idnt] = []
    yrly_rmse[idnt] = []
    yrly_r2ss[idnt] = []
    cmpt[idnt] = []
    
    for yr in all_yrs:
        
        not_nans_Z = ~ np.isnan(Z['HYY_EDDY233.NEE']).values
        #yr_idx = data.index.year == yr
        yr_idx = Z.index.year == yr
        
        completeness = (yr_idx & not_nans_Z).sum() / (365.24*4) * 100
        
        
        obs = Z['HYY_EDDY233.NEE'][yr_idx & not_nans_Z]
        cor = []; r2s = []; rms = []
        for repeat in np.arange(n_repeats).astype(str):
            cor.append(fcts.calc_corr(obs, Z['fcs_'+repeat+'_100'][yr_idx & not_nans_Z]))
            r2s.append(fcts.calc_r2ss(obs, Z['fcs_'+repeat+'_100'][yr_idx & not_nans_Z]))
            rms.append(fcts.calc_rmse(obs, Z['fcs_'+repeat+'_100'][yr_idx & not_nans_Z]))
        
        
        '''
        obs = data['Observed CO2 flux, 6-hourly'  ][yr_idx & not_nans]
        grb = data['GB, random sampling, 6-hourly'][yr_idx & not_nans]
        rfr = data['RF, random sampling, 6-hourly'][yr_idx & not_nans]
        _, bts_rms_grb = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        _, bts_cor_grb = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        _, bts_r2s_grb = fcts.calc_bootstrap(grb.values,obs.values,fcts.calc_r2ss, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        _, bts_rms_rfr = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        _, bts_cor_rfr = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        _, bts_r2s_rfr = fcts.calc_bootstrap(rfr.values,obs.values,fcts.calc_r2ss, [0.5, 2.5, 50, 97.5, 99.5], 1000)
        boot_rms_grb.append(bts_rms_grb)
        boot_cor_grb.append(bts_cor_grb)
        boot_r2s_grb.append(bts_r2s_grb)
        boot_rms_rfr.append(bts_rms_rfr)
        boot_cor_rfr.append(bts_cor_rfr)
        boot_r2s_rfr.append(bts_r2s_rfr)
        '''
        
        yrly_corr[idnt].append(cor) 
        yrly_rmse[idnt].append(rms)
        yrly_r2ss[idnt].append(r2s)
        
        cmpt[idnt].append(completeness.round(1))
        print(yr, completeness.round(1), 'R2SS median:', np.median(r2s))










from matplotlib.gridspec import GridSpec
import matplotlib as mpl

fig=plt.figure(figsize=(16,12))

gs=GridSpec(3,3) # 3 rows, 3 columns

#ax1=fig.add_subplot(gs[0,0]) # First row, first column
#ax2=fig.add_subplot(gs[1,0]) # Second row, first column
#ax3=fig.add_subplot(gs[0,1:]) # First row, second column
#ax4=fig.add_subplot(gs[1,1:]) # Second row, second column
ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[1,0]) # Second row, first column
ax3=fig.add_subplot(gs[2,0]) # Third row, first column
ax4=fig.add_subplot(gs[0,1:]) # First row, second column
ax5=fig.add_subplot(gs[1,1:]) # Second row, second column
ax6=fig.add_subplot(gs[2,1:]) # Third row, second column


sns.set_theme(style="white")
sns.heatmap(mndi_rmse['GB_shuffle=True'], cmap='Blues', annot=True, ax=ax1)
sns.heatmap(mndi_corr['GB_shuffle=True'], cmap='Reds_r', annot=True, ax=ax2)
sns.heatmap(mndi_r2ss['GB_shuffle=True'], cmap='Greens_r', annot=True, ax=ax3)
ax1.set_title('RMSE, GB'); ax1.set_xlabel('Hour');  ax1.set_ylabel('Month'); 
ax2.set_title('CORR, GB'); ax2.set_xlabel('Hour');  ax2.set_ylabel('Month'); 
ax3.set_title('R2SC, GB'); ax3.set_xlabel('Hour');  ax3.set_ylabel('Month'); 

sns.set_theme(style="whitegrid")


sel = np.array(cmpt['GB_shuffle=True']) > 30 
nan = ~sel


to_plot1 = pd.DataFrame(np.array(yrly_rmse['RF_shuffle=True']).T, columns=all_yrs); to_plot1[to_plot1.columns[nan]] = np.nan
to_plot1 = pd.melt(to_plot1, var_name='Year', value_name='RMSE'); to_plot1['Model'] = 'RF'
to_plot2 = pd.DataFrame(np.array(yrly_rmse['GB_shuffle=True']).T, columns=all_yrs); to_plot2[to_plot2.columns[nan]] = np.nan
to_plot2 = pd.melt(to_plot2, var_name='Year', value_name='RMSE'); to_plot2['Model'] = 'GB'
to_plot = pd.concat([to_plot1, to_plot2], ignore_index=True, sort=False)

f = sns.boxplot(data=to_plot, x='Year', y='RMSE', hue='Model', ax=ax4, width=0.7, color='b', showfliers=False, whis=100, linewidth=1 ) 
print(to_plot.groupby('Year').median())

ax4.legend(loc='upper left'); ax4.set_xlabel('')
ax4.set_xticks(ax4.get_xticks()+.5, minor=True)
ax4.yaxis.grid(False); ax4.xaxis.grid(True, which='minor') 



to_plot1 = pd.DataFrame(np.array(yrly_corr['RF_shuffle=True']).T, columns=all_yrs); to_plot1[to_plot1.columns[nan]] = np.nan
to_plot1 = pd.melt(to_plot1, var_name='Year', value_name='CORR'); to_plot1['Model'] = 'RF'
to_plot2 = pd.DataFrame(np.array(yrly_corr['GB_shuffle=True']).T, columns=all_yrs); to_plot2[to_plot2.columns[nan]] = np.nan
to_plot2 = pd.melt(to_plot2, var_name='Year', value_name='CORR'); to_plot2['Model'] = 'GB'
to_plot = pd.concat([to_plot1, to_plot2], ignore_index=True, sort=False)

g = sns.boxplot(data=to_plot, x='Year', y='CORR', hue='Model', ax=ax5, width=0.7, color='r', showfliers=False, whis=100, linewidth=1 ) 
print(to_plot.groupby('Year').median())

ax5.legend(loc='upper left'); ax5.set_xlabel('')
ax5.set_xticks(ax5.get_xticks()+.5, minor=True)
ax5.yaxis.grid(False); ax5.xaxis.grid(True, which='minor') 


to_plot1 = pd.DataFrame(np.array(yrly_r2ss['RF_shuffle=True']).T, columns=all_yrs); to_plot1[to_plot1.columns[nan]] = np.nan
to_plot1 = pd.melt(to_plot1, var_name='Year', value_name='R2SC'); to_plot1['Model'] = 'RF'
to_plot2 = pd.DataFrame(np.array(yrly_r2ss['GB_shuffle=True']).T, columns=all_yrs); to_plot2[to_plot2.columns[nan]] = np.nan
to_plot2 = pd.melt(to_plot2, var_name='Year', value_name='R2SC'); to_plot2['Model'] = 'GB'
to_plot = pd.concat([to_plot1, to_plot2], ignore_index=True, sort=False)

h = sns.boxplot(data=to_plot, x='Year', y='R2SC', hue='Model', ax=ax6, width=0.7, color='g', showfliers=False, whis=100, linewidth=1 ) 
print(to_plot.groupby('Year').median())

ax6.legend(loc='upper left'); ax6.set_xlabel('')
ax6.set_xticks(ax6.get_xticks()+.5, minor=True)
ax6.yaxis.grid(False); ax6.xaxis.grid(True, which='minor') 


#labels = h.get_xticklabels(); h.set_xticklabels(labels, rotation=30)


#ax4.set_title('CORR'); 
ax6.legend(loc='upper left'); ax6.set_xlabel('')
ax6.set_xticks(ax6.get_xticks()+.5, minor=True)
ax6.yaxis.grid(False); ax6.xaxis.grid(True, which='minor') 

labels = f.get_xticklabels(); f.set_xticklabels(labels, rotation=30)
labels = g.get_xticklabels(); g.set_xticklabels(labels, rotation=30)
labels = h.get_xticklabels(); h.set_xticklabels(labels, rotation=30)


plt.figtext(0.02,0.94,'(a)',fontsize=25,fontstyle='italic',fontweight='bold')
plt.figtext(0.34,0.94,'(b)',fontsize=25,fontstyle='italic',fontweight='bold')


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





to_plot = [ data.iloc[not_nans][['Observed CO2 flux, 6-hourly', 'GB, random sampling, 6-hourly']],
            data.iloc[not_nans][['Observed CO2 flux, 6-hourly', 'RF, random sampling, 6-hourly']], 
            data.iloc[not_nans][['Observed CO2 flux, weekly',   'GB, random sampling, weekly']],
            data.iloc[not_nans][['Observed CO2 flux, weekly',   'RF, random sampling, weekly']] ]



sns.set_theme(style="ticks")
f, axes = plt.subplots(2,2, figsize=(10,10))#, sharex=True, sharey=True)
for ax, pl in zip(axes.ravel(), to_plot):
    
    corr = fcts.calc_corr(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    rmse = fcts.calc_rmse(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    r2ss = fcts.calc_r2ss(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    
    # Bootstrap estimates
    rms_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_rmse, [2.5, 50, 97.5], 1000)
    cor_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_corr, [2.5, 50, 97.5], 1000)
    r2s_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_r2ss, [2.5, 50, 97.5], 1000)
    rms_b = rms_b.round(3); cor_b = cor_b.round(3); r2s_b = r2s_b.round(3); 
    print('RMSE bootstrap for',pl.columns[1],':',rms_b)
    print('CORR bootstrap for',pl.columns[1],':',cor_b)
    print('R2SC bootstrap for',pl.columns[1],':',r2s_b)
    
    vmin, vmax = np.ravel(pl).min()-0.5, np.ravel(pl).max()+0.5
    bin_edges = np.linspace(vmin, vmax, 50)
    ax.hist2d(pl[pl.columns[0]], pl[pl.columns[1]], bins=bin_edges, cmap='YlGnBu_r', norm=LogNorm(1))
    
    ax.plot([vmin, vmax], [vmin, vmax], color='b', linestyle="--")
    ax.axhline(0, color='b', linestyle="--")
    ax.axvline(0, color='b', linestyle="--")
    
    ax.set_xlim([vmin, vmax]); ax.set_ylim([vmin, vmax])
    ax.set_xlabel(pl.columns[0]); ax.set_ylabel(pl.columns[1])
    
    ax.text(0.1,0.85,'R2SC: '+str(r2s_b[1])+'\nCORR: '+str(cor_b[1])+'\nRMSE: '+str(rms_b[1]), transform=ax.transAxes)
    



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




