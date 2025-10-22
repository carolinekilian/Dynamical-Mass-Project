import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import matplotlib as mpl

params = {'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'font.family': 'serif',
          'axes.labelsize': 14
          }

mpl.rcParams.update(params)

filename = 'HD131488_corner.pdf'

walkers=22
burnin = 500

names=[r'log(M$_{star}$) (log(M$_{\rm \odot}$))', r'$incl$ ($^{\circ}$)', r'$R_{c}$ (au)', r'$pp$', r'PA ($^{\circ}$)', r'log(M$_{disk}$)',r'$R_\mathrm{in}$ (au)',r'$\Delta \alpha$ (")',r'$\Delta \delta $ (")',r'$v_\mathrm{sys}$ (km/s)','T_atm','Tatm']
# ,logmass_stell,incl,Rc,pp,pa,logmass,Rin,xoff,yoff,vsys,tatm,lnprobs

def corner_plot(csvfile='Jul1_2025_MCMC.csv', burnin = burnin):

    df = pd.read_csv(csvfile)

    #sigma = 0.99999
    sigma = 0.999999426697
    index = df.lnprobs.idxmax()
    best = df.lnprobs[index]
    crit = 1 - sigma
    var = - np.log(crit) * 2

    df = df.drop(df.index[0:burnin * walkers - 1])
    
    df = df.where(np.abs(df.lnprobs - best) <= var)
    df = df.dropna()

    df = df.drop(axis=1, labels='Unnamed: 0')
    df = df.drop(axis=1, labels='lnprobs')

    fig = corner.corner(df, color='k',labels=names, max_n_ticks=2,label_kwargs=dict(fontsize=10), quantiles=[0.16, 0.5, 0.84],show_titles=True)
    fig.suptitle('MCMC Corner Plot for HD131488', fontsize=16)
    plt.savefig(filename, dpi=503)

corner_plot()
plt.show()

plt.savefig(filename)


