import matplotlib.pyplot as plt
import pandas as pd
from p0 import DISKNAME, IMFILE, init_params, DISTANCE_PC, DMR_OUTPUTS, DMR_PREFIX, mcmc_hyper_params, MCMC_OUTPUTS, CSV_FILE
from dmr import calBase
from datetime import date 

current_date=date.today()
def preprocess_specfigfile(file):
    df=pd.read_csv(file, skiprows=10,skipfooter=6,delim_whitespace=True,index_col=0, engine='python')
    df['Npoints']=df['Flux.1']
    df_new=df.drop('Flux.1',axis=1)
    df_new.columns=["Relativisti", "Flux", "PBCFlux", "Npoints"]
    df_new.reset_index(inplace=True)
    return df_new
def make_spectrum_plot(label):
    if label=='MCMC':
       df=pd.read_csv(f"{MCMC_OUTPUTS}/{CSV_FILE}")
       nsteps=str(len(df)//mcmc_hyper_params['nwalkers'])
    else: 
       nsteps=''

    DMR_MIRIAD_PREFIX=f'{DMR_PREFIX}_MIRIAD'
    data_table = preprocess_specfigfile(f'{DMR_OUTPUTS}/{DMR_MIRIAD_PREFIX}_data_spec.txt')
    model_table = preprocess_specfigfile(f'{DMR_OUTPUTS}/{DMR_MIRIAD_PREFIX}_model_spec.txt')
    resid_table = preprocess_specfigfile(f'{DMR_OUTPUTS}/{DMR_MIRIAD_PREFIX}_resid_spec.txt')
    basics=calBase()

    print(len(basics['obsv']), len(data_table['Flux']), len(model_table['Flux']), len(resid_table['Flux']), basics['chanstep'])

    plt.clf()
    plt.plot(basics['obsv'],basics['obsv']*0.0,'k')
    plt.step(basics['obsv'],data_table['Flux']*1e3,label='Data')
    plt.step(basics['obsv'],model_table['Flux']*1e3,label='Model',linestyle='dashed')
    plt.step(basics['obsv'],resid_table['Flux']*1e3,label='Residuals',linestyle='dotted')
    plt.xlabel(r'V$_\mathrm{LSR}$ (km/s)')
    plt.ylabel(r'F$_\nu$ (mJy)')
    plt.title(f"{DISKNAME} {label} nsteps={nsteps} {current_date}")
    plt.xlim(min(basics['obsv']),max(basics['obsv']))
    plt.legend(loc='upper right')
    plt.savefig(f'{DMR_OUTPUTS}/spec_{label}_nsteps={nsteps}_{current_date}.pdf')
    plt.show()

if __name__=="__main__":
   make_spectrum_plot(label='')
