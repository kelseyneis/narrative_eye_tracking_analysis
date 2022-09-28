import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import mpld3
import numpy as np

import statistics
from math import sqrt

# credit: https://stackoverflow.com/questions/59747313/how-to-plot-confidence-interval-in-python
def plot_ci(x, values, z=1.96, horizontal_line_width=0.25, congruent=True, coords=(0,0)):
    if not values:
        return
    mean = statistics.mean(values)
    if len(values) > 2:
        stdev = statistics.stdev(values)
    else:
        stdev = 1
    confidence_interval = z * stdev / sqrt(len(values))
    x = np.asarray(x)
    color = 'blue' if congruent else 'red'
    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    ix, iy = coords
    ax[ix][iy].plot([x, x], [top, bottom], color=color, alpha=0.5)
    ax[ix][iy].plot([left, right], [top, top], color=color, alpha=0.5)
    ax[ix][iy].plot([left, right], [bottom, bottom], color=color)

    return mean, confidence_interval

def plot_bar(x, values, congruent, coords, err=None):
    x = np.asarray(x)
    width = 0.4
    offset = width/2 if congruent else -width/2
    color = 'blue' if congruent else 'red'
    rects = ax[coords[0]][coords[1]].bar(x + offset, values, width, color=color, yerr=err, alpha=0.75)
    return rects

df = pd.read_csv('./RichmondKinney_both_IA_Richmond.txt', delimiter='\t')

##con = t.loc[t['congruent']]
##inc = t.loc[~t['congruent']]
##
##for i in range(1, num_ias):
##    ia = t.loc[t['IA_ID'] == i]
##    con = ia.loc[ia['congruent'] == True]
##    inc = ia.loc[ia['congruent'] == False]
##
##    # do stats for diff attributes...
##    # IA_FIRST_FIXATION_DURATION IA_AVERAGE_FIX_PUPIL_SIZE	IA_FIRST_FIXATION_DURATION	IA_FIRST_FIXATION_TIME
##    # IA_FIRST_RUN_DWELL_TIME	IA_FIRST_RUN_FIXATION_%	IA_FIRST_RUN_FIXATION_COUNT	IA_ID	IA_REGRESSION_IN_COUNT
##    # IA_REGRESSION_OUT_COUNT	IA_REGRESSION_OUT_FULL_COUNT	IA_REGRESSION_PATH_DURATION	IA_SPILLOVER IA_DWELL_TIME
##    att = 'IA_DWELL_TIME'
##    con = con[att].tolist()
##    inc = inc[att].tolist()
##    if len(con) == 0 and len(inc) == 0:
##        continue
##
##    if t[att].dtype == object:
##        con = [float(x) for x in con if x.isnumeric()]
##        inc = [float(x) for x in inc if x.isnumeric()]
##    # fancy footwork to get IA num from IA id...
##    if i == 1:
##        x_coord = 1
##    else:
##        x_coord = 29 - (num_ias - i) + 1 # how many away from the end
##        
##    print(x_coord)
##    #rects1 = plot_bar(x_coord, con, congruent=True, label=labels[x_coord - 1])
##
##    #rects2 = plot_bar(x_coord, inc, congruent=False, label=labels[x_coord - 1])
##    con_av[x_coord - 1] = sum(con)/len(con) if len(con) else 0
##    inc_av[x_coord - 1] = sum(inc)/len(con) if len(con) else 0
##
##    #test = stats.ttest_ind(con, inc, axis=0, alternative='less')
##    #if test.pvalue < 0.1:
##ax.set_title(att)
##bars1 = plot_bar([i for i in range(0, 29)], con_av, congruent=True, label='')
##bars2 = plot_bar([i for i in range(0, 29)], inc_av, congruent=False, label='')
##
##for bars in bars1, bars2:
##    for i, bar in enumerate(bars.get_children()):
##        tooltip = mpld3.plugins.LineLabelTooltip(bar, label=labels[i])
##        mpld3.plugins.connect(plt.gcf(), tooltip)
##
##
##mpld3.save_html(fig, 'fig.html')

def get_trial(df, trial):
    return df.loc[df['id'] == trial]


# df sandboxing
cols = ['IA_FIRST_FIXATION_DURATION',
        'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
        'IA_REGRESSION_OUT_FULL_COUNT',
        'IA_DWELL_TIME', 'IA_FIRST_RUN_DWELL_TIME', 'IA_SPILLOVER', ]
# IA_SPILLOVER
# IA_REGRESSION_PATH_DURATION
for col in cols:
     df[col] = pd.to_numeric(df[col], errors='coerce')

# add spillover col
df['IA_FIRST_RUN_PLUS_SPILLOVER'] = df[['IA_FIRST_RUN_DWELL_TIME', 'IA_SPILLOVER']].sum(axis=1)
cols.append('IA_FIRST_RUN_PLUS_SPILLOVER')
cols.remove('IA_SPILLOVER')

df2 = df.fillna(0)
con = df2.loc[df2['congruent']]
con.sort_values(by='IA_ID')
inc = df2.loc[~df2['congruent']]
inc.sort_values(by='IA_ID')
agg_dict = {'IA_LABEL': 'first'}
for col in cols:
    agg_dict[col] = 'mean'
c_m = con.groupby(['id', 'IA_ID'], as_index=False).agg(agg_dict)
i_m = inc.groupby(['id', 'IA_ID'], as_index=False).agg(agg_dict)

for col in cols:
    agg_dict[col] = 'std'
c_sd = con.groupby(['id', 'IA_ID'], as_index=False).agg(agg_dict)
i_sd = inc.groupby(['id', 'IA_ID'], as_index=False).agg(agg_dict)

for trial in range(1, 21):
    fig, ax = plt.subplots(4, 3)
    con_means= get_trial(c_m, trial)
    inc_means = get_trial(i_m, trial)
    con_std = get_trial(c_sd, trial)
    inc_std = get_trial(i_sd, trial)
    for ind, col in enumerate(cols):
        plot_coords = (ind % 4, ind % 3)  #x,y coords for which subplot to draw on
        ax[plot_coords[0]][plot_coords[1]].set_title('mean ' + col)
        
        congruents = con_means[col].tolist()
        incongruents = inc_means[col].tolist()
        labels = con_means['IA_LABEL'].tolist()
    
        # bars = plot_bar([i for i in range(len(congruents))], congruents, congruent=True, coords=plot_coords, err=con_std[col].tolist())
        bars = plot_bar([i for i in range(len(congruents))], congruents, congruent=True, coords=plot_coords)
        for bar in bars:
            for i, bar in enumerate(bars.get_children()):
                tooltip = mpld3.plugins.LineLabelTooltip(bar, label=labels[i])
                mpld3.plugins.connect(plt.gcf(), tooltip)
                
        labels = inc_means['IA_LABEL'].tolist()
        # bars = plot_bar([i for i in range(len(incongruents))], incongruents, congruent=False, coords=plot_coords, err=inc_std[col].tolist())
        bars = plot_bar([i for i in range(len(incongruents))], incongruents, congruent=False, coords=plot_coords)
        for bar in bars:
            for i, bar in enumerate(bars.get_children()):
                tooltip = mpld3.plugins.LineLabelTooltip(bar, label=labels[i])
                mpld3.plugins.connect(plt.gcf(), tooltip)
    fig.tight_layout()
    mpld3.save_html(fig, 'item_' + str(trial) + '.html')
    
