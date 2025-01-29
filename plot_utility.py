# select one cml:
cml_k = '16'
my_cml = cml_set.sel(cml_id = cml_k)        # .sel is not int indexing, but selecting specific 'label' 
my_ref = ref_set.sel(cml_id = cml_k)

# shaded refernece wet periods from Pycomlink
# set first and last value with zero for correct plotting
my_ref['ref_wd'][0] = False
#my_ref['ref_wd'][-1] = False

# setup figure
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,4))
#ax1 = axs[0].twiny()
#fig.tight_layout()

# plot TRSL
my_cml.trsl.plot.line(x='time', ax=axs[0], label = 'TL');
# plot Rain rate 
my_ref.rain.plot.line(x='time', ax=axs[1], label = 'TL');

# plot real bool wet/dry with 5min precission
wet_start = np.roll(my_ref.ref_wd, -1) & ~my_ref.ref_wd
wet_end = np.roll(my_ref.ref_wd, 1) & ~my_ref.ref_wd
for wet_start_i, wet_end_i in zip(
    wet_start.values.nonzero()[0],
    wet_end.values.nonzero()[0],
):
    axs[1].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html
    axs[0].axvspan(my_ref.time.values[wet_start_i], my_ref.ref_wd.time.values[wet_end_i], color='b', alpha=0.2, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html

# plot trsl gaps
my_cml['trsl_gap'][0] = False
my_cml['trsl_gap'][-1] = False

gap_start = np.roll(my_cml.trsl_gap, -1) & ~my_cml.trsl_gap
gap_end = np.roll(my_cml.trsl_gap, 1) & ~my_cml.trsl_gap
for gap_start_i, gap_end_i in zip(
    gap_start.values.nonzero()[0],
    gap_end.values.nonzero()[0],
):
    axs[0].axvspan(my_cml.time.values[gap_start_i], my_cml.time.values[gap_end_i], color='r', alpha=0.7, linewidth=0); # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axvspan.html
   

# axes limits source: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlim.html
axs[1].set_xlim(my_cml.time.values[0], my_cml.time.values[-1])
axs[0].set_xlabel('')
axs[1].set_title("")

fig.savefig('cml_aligned_ds.svg')