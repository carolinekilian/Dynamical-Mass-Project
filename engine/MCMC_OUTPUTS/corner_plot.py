import os 
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import corner
from datetime import date 
current_date=date.today()
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from p0 import mcmc_hyper_params, CSV_FILE, DISKNAME, name_map, MCMC_OUTPUTS

CSV_PATH = f"{MCMC_OUTPUTS}/{CSV_FILE}"
skipsteps = 0

params = {
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif',
    'axes.labelsize': 14
}
mpl.rcParams.update(params)

nwalkers = mcmc_hyper_params['nwalkers']
skiprows = skipsteps * nwalkers
names = [col for col in pd.read_csv(CSV_PATH).columns[1:-1]]
latex_names = [name_map[name] for name in names]

def corner_plot(csvfile, skiprows=skiprows):
    df = pd.read_csv(csvfile)
    df = df.drop(columns=['Unnamed: 0', 'lnprobs'])
    df = df.iloc[skiprows:]
    
    nsteps = len(df) // nwalkers
    print(f"Plotting corner for {DISKNAME} with {nsteps} steps per walker")
    
    fig = corner.corner(
        df,
        color='k',
        labels=latex_names,
        max_n_ticks=2,
        label_kwargs=dict(fontsize=10),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True
    )
    
    fig.suptitle(f'{DISKNAME} Corner Plot: nsteps={nsteps} {current_date}', fontsize=16)
    filename = f'{MCMC_OUTPUTS}/{DISKNAME}_corner_plot_nsteps={nsteps}_{current_date}.png'
    plt.savefig(filename, dpi=560)
    plt.show()

corner_plot(CSV_PATH)

