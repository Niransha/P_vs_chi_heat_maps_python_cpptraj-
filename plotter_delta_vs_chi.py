# Imports
from __future__ import division
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy, pylab
import seaborn as sns
import re
import glob
import time
import datetime
#sns.set(style='ticks', palette='Set2')
#%matplotlib inline

sns.__version__

def make_Ramp( ramp_colors ):
    from colour import Color
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )
    plt.figure( figsize = (15,3))
    plt.imshow( [list(np.arange(0, len( ramp_colors ) , 0.1)) ] , interpolation='nearest', origin='lower', cmap= color_ramp )
    plt.xticks([])
    plt.yticks([])
    return color_ramp

#custom_ramp = make_Ramp( ['#0000ff','#00ffff','#ffff00','#ff0000' ] )
#custom_ramp = make_Ramp( ['#32369c','#00ff00','#ffff00','#ff0000' ] )
#custom_ramp = make_Ramp( ['#00f3ff','#68ff00','#ffbf00','#ff005c' ] )
#custom_ramp = make_Ramp( ['#00188f','#00bcf2','#00b294','#009e49','#bad80a', '#fff100', '#ff8c00','#e81123','#ec008c', '#68217a' ] )
custom_ramp = make_Ramp( ['#00188f','#00bcf2','#00b294','#009e49','#bad80a', '#fff100', '#ff8c00','#e81123','#ec008c' ] )


import datetime
import glob
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

t_start = datetime.datetime.now()
sdegree = ' (°)'

counter = 0
for filepath in glob.iglob('/mnt/rna/home/nkumarachchi2019/monomer_d1d2chi_QM_done/A/MM_with_v1v3ZEROED_2010chi_P_chi_all_in_one/tmp2.dat'):
    print(filepath)
    df = pd.read_csv(filepath, delim_whitespace=True)

    a = re.split(';|\.|/| |,|_|\t+| +|\*|\n', filepath)
    print(a)
    xname = "delta"
    yname = "chi"
    dimername = "A"
    print(xname, yname, dimername)

    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(5,5))

    ax = sns.kdeplot(data=df, x=df.columns[0], y=df.columns[1], fill=False, thresh=0, levels=20, cmap=custom_ramp, common_norm=True, cbar=True, cbar_kws={'format': '%2.1e', 'label': 'kernel density'})
   
    plt.title(f"{yname} vs {xname} - {dimername}", size=20)
    plt.xlabel(xname + str(sdegree), fontsize=20)
    plt.ylabel(yname + str(sdegree), fontsize=20)
    plt.xlim([50, 180])
    plt.ylim([0, 361])

    # Custom ticks and formatting
    ax.set_yticks(np.arange(0, 361, 30))
    ax.set_xticks(np.arange(50, 181, 30))
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(3)

    plt.savefig(f'/mnt/rna/home/nkumarachchi2019/monomer_d1d2chi_QM_done/A/MM_with_v1v3ZEROED_2010chi_P_chi_all_in_one/plot.png')

    counter += 1
    print(counter)

t_end = datetime.datetime.now()
elapsedTime = (t_end - t_start)

print(elapsedTime.total_seconds())

