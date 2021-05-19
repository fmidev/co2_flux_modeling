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
import cartopy.crs as ccrs
import cartopy.feature as cfeature





'''
code_dir='/users/kamarain/ATMDP-003/'
data_dir='/fmi/scratch/project_2002138/ATMDP-003/'
rslt_dir='/fmi/scratch/project_2002138/ATMDP-003/'

era5_dir='/fmi/scratch/project_2002138/ERA-5_1p0deg/'
'''



code_dir='/path/to/code/'
data_dir='/path/to/data/'
rslt_dir='/path/to/results/'

era5_dir='/path/to/ERA5_data/'



# Read own functions
sys.path.append(code_dir)
import functions as fcts



all_yrs   = np.arange(1996,2019).astype(int)






# Target data 
obs = pd.read_csv(data_dir+'smeardata_20210224_set1.csv')
obs.index = pd.to_datetime(obs[['Year','Month','Day','Hour','Minute','Second']])
obs.index.name = 'time'

vrb = obs.columns[0]
obs = obs[[vrb]].resample('6H').agg(pd.Series.mean, skipna=False)



obs.plot(); plt.show()





















#bbox = [ 61.84741, 24.29477]

# Read ERA5 predictor data
era5_data = xr.open_dataset(data_dir+'era5_preprocessed.nc').sel(
    time=slice('1996-01-01', '2018-12-31'), 
    lat=slice(60,64),
    lon=slice(22,26)
    )


vrbs = list(era5_data.data_vars)
vrbs.sort()
















# Read modeled results and the model
Z = pd.read_csv(data_dir+'obs_mod_co2.csv', index_col=0)
Z.index = pd.to_datetime(Z.index)
Z['fcs'] = Z['fcs_100'].copy(deep=True)

models = []
mdl_files = np.sort(glob.glob(data_dir+'model_*_for_co2.json'))
for mdl in mdl_files:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(mdl)
    models.append(xgb_model)









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

















# Data selection
data = Z.copy(deep=True)
data = data.rename(columns={vrb:    'Observed CO2 flux, 6-hourly', 
                            'fcs':  'Predicted CO2 flux, 6-hourly'})

#from scipy import ndimage
not_nans_6h = ~ np.isnan(data['Observed CO2 flux, 6-hourly']).values
not_nans_wk = not_nans_6h # ndimage.binary_erosion(not_nans_6h, structure=np.ones(28)).astype(not_nans_6h.dtype)


data['Observed CO2 flux, weekly']  = fcts.kz_filter(data['Observed CO2 flux, 6-hourly'].loc[not_nans_6h], 28, 1)
data['Predicted CO2 flux, weekly'] = fcts.kz_filter(data['Predicted CO2 flux, 6-hourly'].loc[not_nans_6h], 28, 1)

data['Observed CO2 flux, weekly'].loc[~ not_nans_wk] = np.nan









# Amount of fitting data
corr_subs = []
rmse_subs = []

percentages = np.arange(10,101,10)

for pr in percentages:
    obs = data['Observed CO2 flux, 6-hourly'].values
    fcs = data['fcs_'+str(pr)]
    
    cor = fcts.calc_corr(obs, fcs)
    rms = fcts.calc_rmse(obs, fcs)
    
    corr_subs.append(cor)
    rmse_subs.append(rms)






sns.set_theme(style="white")
f, ax1 = plt.subplots(1,1,figsize=(5,4))
ax2 = ax1.twinx()

plt.title('Dependency of RMSE and CORR\non the amount of fitting data')

ax1.plot(percentages, corr_subs, lw=1.5, c='b', label='CORR', alpha=1)
ax2.plot(percentages, rmse_subs, lw=1.5, c='r', label='RMSE', alpha=1)

ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')

ax1.set_ylabel('CORR', color='b')
ax2.set_ylabel('RMSE', color='r')

ax1.set_xlabel('Amount [%]')

plt.tight_layout()
f.savefig(rslt_dir+'fig_corr_rmse_percentages.pdf')
f.savefig(rslt_dir+'fig_corr_rmse_percentages.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')








# Plot climatology
obs_ann = pd.DataFrame(columns=range(1997, 2020))
for year in range(1997, 2020): 
    idx = obs.index.year == year
    len_year = len(obs.loc[idx].values[:1461])
    obs_ann[year] = np.squeeze(obs.loc[idx].values[:1460])
    print(year, len_year)


idx = obs.index.year == 1998
obs_ann.index = obs.loc[idx].index.values




clim_6h = obs_ann.mean(axis=1)
clim_1d = fcts.kz_filter(obs_ann, 4, 1).mean(axis=1)
clim_1w = fcts.kz_filter(obs_ann, 28, 1).mean(axis=1)
clim_1m = fcts.kz_filter(obs_ann, 112, 1).mean(axis=1)


import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import matplotlib.dates as mdates



sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(12, 3))

clim_6h.plot(lw=0.5, c='k', label='6H-climatology', ax=ax, alpha=0.8)
clim_1w.plot(lw=2.5, c='k', label='1W-climatology', ax=ax, alpha=1)




ax.set_ylabel('µmol m⁻² s⁻¹'); ax.set_xlabel('Time of year')
ax.set_title('Climatology of measured CO₂ flux')
ax.axhline(0,c='b', label='Neutral flux'); 
ax.axhline(np.nanmean(obs),c='Orange', label='Mean flux'); 
ax.legend(loc='lower left'); plt.tight_layout()
f.savefig(rslt_dir+'fig_climatology.pdf')
f.savefig(rslt_dir+'fig_climatology.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')











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
f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_gain.pdf')
f.savefig(rslt_dir+'fig_feat_importance_PREDICTORS_gain.png', dpi=200)
plt.show()



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
f.savefig(rslt_dir+'fig_feat_importance_MEAN_gain.pdf')
f.savefig(rslt_dir+'fig_feat_importance_MEAN_gain.png', dpi=200)
plt.show()




mean_scores = all_scores.groupby('Grid cell').mean().sort_values('Mean gain')

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(1,1,figsize=(8, 8))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')
ax.set_title('Mean gain over models, variables and lags')
plt.tight_layout()
f.savefig(rslt_dir+'fig_feat_importance_GRIDCELL_gain.pdf')
f.savefig(rslt_dir+'fig_feat_importance_GRIDCELL_gain.png', dpi=200)
plt.show()



mean_scores = all_scores.groupby('Lag').mean().sort_values('Mean gain')

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(1,1,figsize=(8, 8))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')
ax.set_title('Mean gain over models, variables, and grid cells')
plt.tight_layout()
f.savefig(rslt_dir+'fig_feat_importance_LAG_gain.pdf')
f.savefig(rslt_dir+'fig_feat_importance_LAG_gain.png', dpi=200)
plt.show()




mean_scores = all_scores.groupby(['Grid cell', 'Lag']).mean().sort_values('Mean gain')

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(1,1,figsize=(4, 12))
mean_scores.plot.barh(ax=ax, legend=False)
ax.set_xlabel('F score')

ax.set_title('Mean gain over models and variables, \ngrouped by (grid cells, lags)')
plt.tight_layout()
ax.set_xscale('log')
ax.tick_params(axis='both', which='major', labelsize=6)
f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_gain.pdf')
f.savefig(rslt_dir+'fig_feat_importance_LAGetGRIDCELL_gain.png', dpi=200)
plt.show()












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
f.savefig(rslt_dir+'fig_timeser1.pdf')
f.savefig(rslt_dir+'fig_timeser1.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')





# Time series, three sample years
sns.set_theme(style="whitegrid")

smpl_yrs = [2003, 1999, 1997] #
smpl_yrs = [2008, 1999, 2001] #
smpl_yrs = [2016, 1999, 2004] #
#smpl_yrs = random.sample(list(all_yrs), 3)

f, ax = plt.subplots(len(smpl_yrs), 1, figsize=(12, 3*len(smpl_yrs)))
linewidths=[0.7, 0.7, 5, 3]
colors=['red', 'k', 'red', 'k']
cols = ['Predicted CO2 flux, 6-hourly', 'Observed CO2 flux, 6-hourly',
        'Predicted CO2 flux, weekly', 'Observed CO2 flux, weekly']
for i,yr in enumerate(smpl_yrs):
    for col, lw, clr in zip(cols, linewidths, colors):
        yr_idx = data.index.year == yr
        if col == 'Observed CO2 flux, 6-hourly': sel = yr_idx
        if col == 'Observed CO2 flux, weekly':   sel = yr_idx #& not_nans
        else: sel = np.ones(data.index.shape, bool)
        
        to_plot = data[col].copy(deep=True) # .iloc[sel].loc[str(yr)+'-01-01':str(yr)+'-12-31']
        to_plot.iloc[~sel] = np.nan
        to_plot.loc[str(yr)+'-01-01':str(yr)+'-12-31'].plot(lw=lw, c=clr, label=col, ax=ax[i], alpha=0.8)
        ax[i].set_xlim(str(yr)+'-01-01', str(yr)+'-12-31')
        ax[i].axhline(0,c='b'); 
        ax[i].set_ylabel('µmol m⁻² s⁻¹'); #ax[i].set_xlabel('Time of year')


ax[0].legend(loc='lower left')
plt.tight_layout(); 
f.savefig(rslt_dir+'fig_timeser2.pdf')
f.savefig(rslt_dir+'fig_timeser2.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')









# Diurnal and monthly decomposition and analysis of CORR and RMSE skill
mndi_c = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
mndi_r = pd.DataFrame(index=np.arange(1,13), columns=(0, 6, 12, 18))
for hr in (0, 6, 12, 18):
    for mn in np.arange(1,13):
        idx = (data.index.hour == hr) & (data.index.month == mn)
        obs = data['Observed CO2 flux, 6-hourly' ][idx & not_nans_6h]
        fcs = data['Predicted CO2 flux, 6-hourly'][idx & not_nans_6h]
        cor_b, _ = fcts.calc_bootstrap(fcs.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 10000)
        rms_b, _ = fcts.calc_bootstrap(fcs.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 10000)
        print(mn, hr, cor_b, rms_b)
        mndi_c.loc[mn,hr] = cor_b[2]
        mndi_r.loc[mn,hr] = rms_b[2]


mndi_c = mndi_c.astype(float)
mndi_r = mndi_r.astype(float)




# Evolution of CORR and RMSE skill over time
rmse = []
corr = []
boot_rms = []
boot_cor = []
cmpt = []

#to_plot = data[['Observed CO2 flux, 6-hourly','Observed CO2 flux, weekly']]
for yr in all_yrs:
    yr_idx = data.index.year == yr
    completeness = (yr_idx & not_nans_6h).sum() / (365.24*4) * 100
    obs = data['Observed CO2 flux, 6-hourly' ][yr_idx & not_nans_6h]
    fcs = data['Predicted CO2 flux, 6-hourly'][yr_idx & not_nans_6h]
    rms_b, bts_rms = fcts.calc_bootstrap(fcs.values,obs.values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 10000)
    cor_b, bts_cor = fcts.calc_bootstrap(fcs.values,obs.values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 10000)
    boot_rms.append(bts_rms)
    boot_cor.append(bts_cor)
    cmpt.append(completeness.round(1))
    print(yr, completeness.round(1), 'RMSE median:',rms_b[2].round(3),'CORR median:',cor_b[2].round(3) )








from matplotlib.gridspec import GridSpec


fig=plt.figure(figsize=(16,8))

gs=GridSpec(2,3) # 2 rows, 3 columns

ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[1,0]) # Second row, first column
ax3=fig.add_subplot(gs[0,1:]) # First row, second column
ax4=fig.add_subplot(gs[1,1:]) # First row, third column


sns.set_theme(style="white")
sns.heatmap(mndi_r, cmap='hot_r', annot=True, ax=ax1)
sns.heatmap(mndi_c, cmap='YlGnBu_r', annot=True, ax=ax2)
ax1.set_title('RMSE'); ax1.set_xlabel('Hour');  ax1.set_ylabel('Month'); 
ax2.set_title('CORR'); ax2.set_xlabel('Hour');  ax2.set_ylabel('Month'); 


sns.set_theme(style="whitegrid")


sel = np.array(cmpt) > 60 
nan = ~sel


to_plot = pd.DataFrame(np.array(boot_rms).T, columns=all_yrs)
to_plot[to_plot.columns[nan]] = np.nan
g = sns.boxenplot(data=to_plot, ax=ax3, width=0.7, color='r', k_depth=10, showfliers=False, linewidth=1 ) # whis=[0, 100], notch=True,)
ax3.set_title('RMSE')
labels = g.get_xticklabels(); g.set_xticklabels(labels, rotation=30)

to_plot = pd.DataFrame(np.array(boot_cor).T, columns=all_yrs)
to_plot[to_plot.columns[nan]] = np.nan
h = sns.boxenplot(data=to_plot, ax=ax4, width=0.7, color='r', k_depth=10, showfliers=False, linewidth=1 ) # whis=[0, 100], notch=True,)
ax4.set_title('CORR')
labels = h.get_xticklabels(); h.set_xticklabels(labels, rotation=30)


plt.figtext(0.02,0.94,'a)',fontsize=25,fontstyle='italic',fontweight='bold')
plt.figtext(0.34,0.94,'b)',fontsize=25,fontstyle='italic',fontweight='bold')


#plt.suptitle('Bootstrap estimates of RMSE and Pearson correlation with 10$^4$ samples, estimated from 6-hourly data')
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90,wspace=0.2,hspace=0.3)
fig.savefig(rslt_dir+'fig_RMSE_time.pdf',)# bbox_inches='tight')
fig.savefig(rslt_dir+'fig_RMSE_time.png', dpi=200,)# bbox_inches='tight')
plt.show(); plt.clf(); plt.close('all')











# Scatter plots
sns.set_theme(style="ticks")
f, axes = plt.subplots(1,2, figsize=(10,5))


for ax in axes:
    ax.axhline(0, linestyle="--")
    ax.axvline(0, linestyle="--")


to_plot = [ data.iloc[not_nans_6h][['Observed CO2 flux, 6-hourly', 'Predicted CO2 flux, 6-hourly']], 
            data.iloc[not_nans_wk][['Observed CO2 flux, weekly',   'Predicted CO2 flux, weekly']] ]

for i, pl in enumerate(to_plot):
    
    corr = fcts.calc_corr(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    rmse = fcts.calc_rmse(pl[pl.columns[0]], pl[pl.columns[1]]).round(3)
    
    # Bootstrap estimates
    rms_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_rmse, [0.5, 2.5, 50, 97.5, 99.5], 10000)
    cor_b, _ = fcts.calc_bootstrap(pl[pl.columns[1]].values,pl[pl.columns[0]].values,fcts.calc_corr, [0.5, 2.5, 50, 97.5, 99.5], 10000)
    print('RMSE bootstrap for',pl.columns[1],':',rms_b)
    print('CORR bootstrap for',pl.columns[1],':',cor_b)
    
    sns.regplot(data=pl, x=pl.columns[0], y=pl.columns[1], marker=',', 
        scatter_kws={'s': 1}, ax=axes[i])
    
    axes[i].text(0.1,0.8,'CORR: '+str(corr)+'\nRMSE: '+str(rmse), transform=axes[i].transAxes)
    




axes[0].set_title('6-hourly CO2 flux [µmol m⁻² s⁻¹]')
axes[1].set_title('Weekly CO2 flux [µmol m⁻² s⁻¹]')

plt.tight_layout(); 
f.savefig(rslt_dir+'fig_scatter.pdf')
f.savefig(rslt_dir+'fig_scatter.png', dpi=200)
plt.show(); plt.clf(); plt.close('all')




