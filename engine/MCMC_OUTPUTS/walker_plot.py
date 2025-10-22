import os 
import sys 
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
current_date=date.today()
# you'll need this in order to import variables from python files in parent direcotries
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from p0 import mcmc_hyper_params, CSV_FILE, MCMC_OUTPUTS

def walker_plot(filename, nwalkers=mcmc_hyper_params['nwalkers'], only_these_columns=[], burnin=0):
        """
           generate walker plot given the MCMC-generated 
           pandas dataframe 
        """
        df=pd.read_csv(filename)
        steps = len(df) // nwalkers        
        print(f'THIS CHAIN RAN FOR {steps} STEPS PER WALKER.')
        columns_to_plot = only_these_columns if only_these_columns else df.columns

        for col in columns_to_plot: 
           
           plt.figure(figsize=(12,4))
           col_data=df[col].values
           
           for n in range(nwalkers):
              # start at index n and take steps of nwalkers until you reach the end 
              w_n = col_data[n::nwalkers]            
              plt.plot(range(len(w_n)), w_n)
	

           plt.title(f'Walker Plot - {col}; nsteps={steps}; {current_date}, {filename}')
           plt.xlabel('Steps')
           plt.ylabel(col)
           plt.savefig(f'{MCMC_OUTPUTS}/walker_plot_{col}_nsteps={steps}_{current_date}.pdf', dpi=500)
           plt.show()

walker_plot(f"{MCMC_OUTPUTS}/{CSV_FILE}")
