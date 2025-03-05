

# # Wet Antenna Attenuation
# For CNN predicted wet/dry periods.  
# Using modules imported from Pycomlink and code based on Pycomlink Tutorials

# #### Reshape dataset back to time domain
# <span style="color:red">Assuming, each reference RADOLAN rain sample taken in time TT is measurement from <TT; TT + 5min> period. </span> This assumption may itroduce up to 5min time shift between reference and CML data (TT measurement contains rather value from <TT-5min; TT>), but doesn't introduce any invasive change to the original data.  
# In future RADOLAN ref values may be shifted -5 minutes in preprocessing.
# 
# Please note that:
# 
# - The wet-dry classification, which is done is a very simple way here, also plays a significant role for rain rates estimation.
# - For a different CMLs, different WAA methods and different parameters might perform best.
# - The radar rainfall reference cannot not be trusted 100% at its 5-minute resolution since the gauge-adjustment is done on hourly basis and then applied to the 5-min data. In addition, the radar reference might have larger uncertainty in regions far away from the gauges that have been used for the hourly adjustment. We have not included this info in our RADKLIM-YW dataset, hence, we have to treat the radar_along_cml equally for all CMLs. But, differences between CML and radar reference do not always mean the CML is wrong.
# 
# Below you can change the CML that is chosen and also play with different WAA parameters. You will find that there is no optimal solution for all CMLs, and that the wet-dry classification has a strong impact on the overall CML rainfall sum.

# In[ ]:


# Reshape dataset back to time domain
trsl_waa = ds.trsl.values.reshape(num_cmls,2,-1)
cnn_waa = np.repeat(ds.cnn_wd.values.reshape(num_cmls,-1),10, axis=1)   # cnn_wd output upsampled to 1min by repeating the values 
time_waa = ds.time.values.reshape(-1)                  # to be dimension again


# In[ ]:


# Create a new xarray Dataset in time domain
# cnn_wd upsampled 10 times
ds_waa = xr.Dataset({
    'trsl': (('cml_id','channel_id', 'time'), trsl_waa),
    'cnn_wd': (('cml_id','time'), cnn_waa) #,    'R_ref': (('cml_id','time'), R_ref_waa)
}, coords={'cml_id': cml_id[:num_cmls],
           'channel_id': np.arange(2),
           'time': time_waa,
           'length':(('cml_id'), length[:num_cmls]),                                  #single value is never a dimension
           'frequency': (('cml_id', 'channel_id'), frequency[:num_cmls]),          # this way coordinate will not become a dimension
           'polarization': (('cml_id', 'channel_id'), polarization[:num_cmls])
})


# #### CML baseline for CNN predicted Wet signal calculated with Pycomlink modul

# In[ ]:


cml_k = '1'


# In[ ]:


ds_waa['baseline'] = pycml.processing.baseline.baseline_constant(
        trsl=ds_waa.trsl,
        wet=ds_waa.cnn_wd,
        n_average_last_dry=10
)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,3))

ds_waa.trsl.sel(cml_id=cml_k, time=slice('2018-05-13T12:00', '2018-05-14T06:00')).plot.line(x='time', alpha=0.5)
plt.gca().set_prop_cycle(None)  # reuse plot colors
ds_waa.baseline.sel(cml_id=cml_k, time=slice('2018-05-13T12:00', '2018-05-14T06:00')).plot.line(x='time');
plt.ylabel('TRSL');


# #### Wet antenna attenuation estimation
# from Pycomlink using Schleiss, Leijnse and Pastorek mathematical models
# 
# 1) Rain rate without WAA comp
# #### <span style="color:red">TODO: </span> Rain rate calculated from both Channels separatelly, Choose one or average

# In[ ]:


# Rain induced attenuation
ds_waa['A_rain'] = ds_waa.trsl - ds_waa.baseline
ds_waa['A_rain'].values[ds_waa.A_rain < 0] = 0

# calculate R using k-R relation without WAA comp.
ds_waa['R'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=ds_waa.A_rain, 
        L_km=ds_waa.length, 
        f_GHz=ds_waa.frequency/1e9, 
        pol=ds_waa.polarization
)


# 2) WAA Schleiss, Leijnse and Pastorek

# In[ ]:


ds_waa['waa_schleiss'] = pycml.processing.wet_antenna.waa_schleiss_2013(
    rsl=ds_waa.trsl, 
    baseline=ds_waa.baseline, 
    wet=ds_waa.cnn_wd, 
    waa_max=2.2, 
    delta_t=1, 
    tau=15,
)

ds_waa['waa_leijnse'] = pycml.processing.wet_antenna.waa_leijnse_2008_from_A_obs(
    A_obs=ds_waa.A_rain,
    f_Hz=ds_waa.frequency,
    pol=ds_waa.polarization,
    L_km=ds_waa.length,
)

ds_waa['waa_pastorek'] = pycml.processing.wet_antenna.waa_pastorek_2021_from_A_obs(
    A_obs=ds_waa.A_rain,
    f_Hz=ds_waa.frequency,
    pol=ds_waa.polarization,
    L_km=ds_waa.length,
    A_max=2.2,
)


# 3. Rain induced ATT and rain rate using k-R relation. For all 3 WAA compensation methods

# In[ ]:


# R for all 3 WAA methods
for waa_method in ['leijnse', 'pastorek', 'schleiss']:
    ds_waa[f'A_rain_{waa_method}'] = ds_waa.A_rain - ds_waa[f'waa_{waa_method}']
    ds_waa[f'A_rain_{waa_method}'] = ds_waa[f'A_rain_{waa_method}'].where(ds_waa[f'A_rain_{waa_method}'] >= 0, 0)    
    
    ds_waa[f'R_{waa_method}'] = pycml.processing.k_R_relation.calc_R_from_A(
        A=ds_waa[f'A_rain_{waa_method}'], 
        L_km=ds_waa.length, 
        f_GHz=ds_waa.frequency/1e9, 
        pol=ds_waa.polarization
    )


# ####  Predicted Rainrates in time domain + Cumulative sum

# In[ ]:


# note: 
#     Reference 1h rainfall rate sampled to 5 min has to be converted to 5min rain rate ... multiplied by 60/5 min
#     CML rain rate will be resampled to 5 min 

# TRSL + rain induced attenuation
fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
plt.sca(axs[0])
ds_waa.trsl.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='TRSL', color='k', zorder=10)
ds_waa.baseline.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='baseline', color='C0')
(ds_waa.baseline + ds_waa.waa_leijnse).sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_leijnse', color='C1')
(ds_waa.baseline + ds_waa.waa_pastorek).sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_pastorek', color='C2')
(ds_waa.baseline + ds_waa.waa_schleiss).sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='baseline + WAA_schleiss', color='C3')
plt.ylabel('total path attenuation in dB')
plt.title(f'cml_id = {ds_waa.cml_id.sel(cml_id=cml_k).values}   length = {ds_waa.length.sel(
    cml_id=cml_k).values.round(decimals=3) } km   frequency = {ds_waa.frequency.sel(cml_id=cml_k).isel(channel_id=0).values/1e9} GHz')
plt.legend()

# Reference RADOLAN rainrate with 
plt.sca(axs[1])
(ref_set_copy.sel(cml_id=cml_k).rain * 12).plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)
ds_waa.R.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='no WAA', color='C0')
ds_waa.R_leijnse.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='with WAA_leijnse', color='C1')
ds_waa.R_pastorek.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='with WAA_pastorek', color='C2')
ds_waa.R_schleiss.sel(cml_id=cml_k).isel(channel_id=0).plot.line(x='time', label='with WAA_schleiss', color='C3')
plt.ylabel('Rain-rate in mm/h')
plt.title('')
plt.legend()

plt.sca(axs[2])
ref_set_copy.sel(cml_id=cml_k).rain.cumsum(dim='time').plot.line(color='k', linewidth=3.0, label='RADKLIM-YW', alpha=0.3)
(ds_waa.R.sel(cml_id=cml_k).isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='no WAA', color='C0')
(ds_waa.R_leijnse.sel(cml_id=cml_k).isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_leijnse', color='C1')
(ds_waa.R_pastorek.sel(cml_id=cml_k).isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_pastorek', color='C2')
(ds_waa.R_schleiss.sel(cml_id=cml_k).isel(channel_id=0)/60).cumsum(dim='time').plot.line(x='time', label='with WAA_schleiss', color='C3')
plt.ylabel('Rainfall sum in mm')
plt.title('')
plt.legend();

axs[1].set_xlim(pd.to_datetime('2018-05-13T18:00'), pd.to_datetime('2018-05-14T04:00'));


# #### Rainrate XY scatterplot
# CML predicted Rainrate vs real Rainrate. Shows spread on different Rainrates

# In[ ]:


def hexbinplot(R_radar_along_cml, R_cml, ax, color='k', title=None, loglog=True):
    R_cml = R_cml.isel(channel_id=0).resample(time='5min').mean().reindex_like(R_radar_along_cml, method=None)
    R_cml.values[R_cml.values < 0] = 0
    #R_cml = R_cml.resample(time='1h').mean()
    #R_radar_along_cml = R_radar_along_cml.resample(time='1h').mean()
    ax.scatter(
        R_radar_along_cml.where(R_radar_along_cml > 0).values,
        R_cml.where(R_cml > 0).values,
        c=color,
        s=10,
        alpha=0.7,
    )
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel('Average radar rain rate along CML in mm/h')
    ax.set_ylabel('CML rain rates in mm/h')


# In[ ]:


ref_R = ref_set_copy.sel(cml_id=cml_k).rain * 12
ds_hex = ds_waa.sel(cml_id=cml_k)


# In[ ]:


fig, axs = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)

hexbinplot(ref_R, ds_hex.R, axs[0], 'C0', 'no WAA')
hexbinplot(ref_R, ds_hex.R_leijnse, axs[1], 'C1', 'WAA Leijnse')
hexbinplot(ref_R, ds_hex.R_pastorek, axs[2], 'C2', 'WAA Pastorek')
hexbinplot(ref_R, ds_hex.R_schleiss, axs[3], 'C3', 'WAA Schleiss')

for ax in axs:
    ax.plot([0.01, 50], [0.01, 50], 'k', alpha=0.3)
    ax.set_xlim(0.05, 50)
    ax.set_ylim(0.05, 50)

fig, axs = plt.subplots(1, 4, figsize=(22, 5), sharex=True, sharey=True)

loglog=False
hexbinplot(ref_R, ds_hex.R, axs[0], 'C0', 'no WAA', loglog=loglog)
hexbinplot(ref_R, ds_hex.R_leijnse, axs[1], 'C1', 'WAA Leijnse', loglog=loglog)
hexbinplot(ref_R, ds_hex.R_pastorek, axs[2], 'C2', 'WAA Pastorek', loglog=loglog)
hexbinplot(ref_R, ds_hex.R_schleiss, axs[3], 'C3', 'WAA Schleiss', loglog=loglog)

for ax in axs:
    ax.plot([0.01, 50], [0.01, 50], 'k', alpha=0.3)
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-0.5, 20)


# #### 5min rainrate dependent Rain accumulation 
# Shows deviation of WAA comp. methods from real Rainrate

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 5))
plt.plot(np.sort(ref_R.values), np.sort(ref_R.values / 12).cumsum(), label='RADKLIM along CML', color='k', alpha=0.3, linewidth=3)

R_cml = ds_hex.R.isel(channel_id=0).resample(time='5min').mean().reindex_like(ref_R, method=None)
plt.plot(np.sort(R_cml.values), np.sort(R_cml.values / 12).cumsum(), label='CML without WAA correction')

R_cml = ds_hex.R_leijnse.isel(channel_id=0).resample(time='5min').mean().reindex_like(ref_R, method=None)
plt.plot(np.sort(R_cml.values), np.sort(R_cml.values / 12).cumsum(), label='CML with Leijnse WAA')

R_cml = ds_hex.R_pastorek.isel(channel_id=0).resample(time='5min').mean().reindex_like(ref_R, method=None)
plt.plot(np.sort(R_cml.values), np.sort(R_cml.values / 12).cumsum(), label='CML with Pastorek WAA')

R_cml = ds_hex.R_schleiss.isel(channel_id=0).resample(time='5min').mean().reindex_like(ref_R, method=None)
plt.plot(np.sort(R_cml.values), np.sort(R_cml.values / 12).cumsum(), label='CML with Schleiss WAA')

plt.xlabel('5-minute rain rate in mm/h')
plt.ylabel('rainfall accumulation in mm')
plt.legend();
#plt.xscale('log')
#plt.xlim(0.1, 50)


# ### ____________________________________________________________________________________________________________

# ### rolling standard deviation classification for reference

# In[ ]:


threshold = 0.8

roll_std_dev = my_cml.trsl.rolling(time=60, center=True).std()
my_cml['rsd_wet'] = my_cml.trsl.rolling(time=60, center=True).std() > threshold


# In[ ]:


my_cml


# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(12,5), sharex=True)

roll_std_dev.plot.line(x='time', ax=axs[0])
axs[0].axhline(threshold, color='k', linestyle='--')

my_cml.trsl.plot.line(x='time', ax=axs[1]);

# Get start and end of dry event
wet_start = np.roll(my_cml.rsd_wet, -1) & ~my_cml.rsd_wet
wet_end = np.roll(my_cml.rsd_wet, 1) & ~my_cml.rsd_wet

# Plot shaded area for each RSD predicted wet event
for wet_start_i, wet_end_i in zip(
    wet_start.isel(channel_id=0).values.nonzero()[0],
    wet_end.isel(channel_id=0).values.nonzero()[0],
):
    axs[1].axvspan(my_cml.time.values[wet_start_i], my_cml.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0)

axs[1].set_title('');

# plot real bool wet/dry with 5min precission
wet_start = np.roll(my_ref.ref_wet_dry, -1) & ~my_ref.ref_wet_dry
wet_end = np.roll(my_ref.ref_wet_dry, 1) & ~my_ref.ref_wet_dry
for wet_start_i, wet_end_i in zip(
    wet_start.values.nonzero()[0],
    wet_end.values.nonzero()[0],
):
    axs[1].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wet_dry.time.values[wet_end_i], color='g', alpha=0.2, linewidth=0) # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html

