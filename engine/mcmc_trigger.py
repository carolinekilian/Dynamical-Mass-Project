from galario.double import sampleImage
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import emcee
import os
import shutil
import sys
from disk import *
from raytrace import *
import time
from schwimmbad import MPIPool
from astropy import units as u
from p0 import init_params, params_uncertainty, DISTANCE_PC, SIMBAD_parallax_arcsec, SIMBAD_parallax_uncertainty_arcsec, IMFILE, DATAFILE, ADDITIONAL_OBS, EXCLUDED_CHANNEL_IDXs, CSV_FILE, mcmc_hyper_params, param_order, param_priors, MCMC_OUTPUTS 
from dmr import calBase, makeGaussian
debug=False

def valid_parameter_names(init_params, 
                          params_uncertainty, 
                          param_priors,
                          required_params=['logmass', 'logmass_stell', 'Rin', 'Rc', 'tatm']):
    intersection_set=set(init_params.keys()) & set(params_uncertainty) & set(param_priors)
    for name in required_params:
        if name not in intersection_set:
           return False, f"{name} not in init_params, params_uncertainty, and param_priors and is required for disk model"
    return True, ''
    
def chiSq(datafile, modfile, dxy, dRA, dDec, pbcor):
    """
    computes the goodness of fit metric, chi-squared, 
    between the model and the data 
    """
	
    mod_arr=[]
    modfile_name=modfile + str('.fits')
    model_fits=fits.open(modfile_name)
    
    model=model_fits[0].data
    model=model.byteswap().newbyteorder().squeeze()
    model=model[:,::-1,:]
    model=model.copy(order='C')
    model_cor=model * pbcor
    model_cor=model_cor.copy(order='C')
    
    model_fits.close()

    chi=0 
    data_vis=fits.open(datafile)
    data_shape=data_vis[0].data['data'].shape
    delta=data_vis[0].header['CDELT4']
    freq0=data_vis[0].header['CRVAL4'] 
    n_spw=data_vis[0].header['NAXIS4'] 
    
    model_vis=np.zeros(data_shape)
    data=data_vis[0].data['data']

    freq_start=freq0 - (n_spw - 1.0) / 2.0 * delta

    for i in range(n_spw):
        if i in EXCLUDED_CHANNEL_IDXs:
           print(f"EXCLUDING CHANNEL {i} FROM CHI SQR")
           continue 
        freq=freq_start + i * delta
        u, v=(data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64)
    	
        model_cor=np.require(model_cor, requirements='C')
        
        vis=sampleImage(model_cor[i,:,:], dxy, u, v, dRA=dRA, dDec=dDec) 
        
        model_vis[:,0,0,0,i,0,0]=vis.real
        model_vis[:,0,0,0,i,1,0]=vis.real
        model_vis[:,0,0,0,i,0,1]=vis.imag
        model_vis[:,0,0,0,i,1,1]=vis.imag

        chi +=((vis.real - data[:,0,0,0,i,0,0])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.imag - data[:,0,0,0,i,0,1])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.real - data[:,0,0,0,i,1,0])**2 * data[:,0,0,0,i,1,2]).sum() + ((vis.imag - data[:,0,0,0,i,1,1])**2 * data[:,0,0,0,i,1,2]).sum()

    
    data_vis.close()
    
    return chi 

def lnprob(p0):
   """ 
   ensures that the sampled parameter values are within physically reasonable 
   bounds. If ALL parameters are physically reasonable, then a disk model is created, 
   raytraced, and compared with the initial data. The goodness of fit between the  
   model and the data is quantitatively measured using the chi-sqr metric. 
   """
   if debug:
      print(f"Sampled values: type: {type(p0)}; {p0}")
   # Sample parallax and compute distance at each MCMC step
   parallax_mean=SIMBAD_parallax_arcsec
   parallax_std=SIMBAD_parallax_uncertainty_arcsec

   parallax=np.random.normal(parallax_mean, parallax_std)
   if parallax <=0:
      print("Invalid parallax sample")
      return -np.inf  # physically invalid

   distance=1.0 / parallax  # parsecs

   # Log sampled values to file
   with open(f"{MCMC_OUTPUTS}/parallax_log.txt", "a") as f:
      f.write(f"parallax={parallax:.6e} arcsec → distance={distance:.2f} pc\n")

   valid_sample_parameters=dict()
   for i,param_name in enumerate(param_order):
       mcmc_parameter_sample=p0[i]
       if mcmc_parameter_sample < param_priors[param_name][0] or mcmc_parameter_sample > param_priors[param_name][1]:
          print(f"{param_name} is out of bounds: sampled_value={mcmc_parameter_sample}; priors: {param_priors[param_name]}")
          return -np.inf
       else:
          valid_sample_parameters[param_name]=mcmc_parameter_sample

   if valid_sample_parameters['Rin'] >= valid_sample_parameters['Rc']:
      print(f"Sampled Rin, Rin={valid_sample_parameters['Rin']}, >= Sampled Critical Radius, Rc={valid_sample_parameters['Rc']}")
      return -np.inf
   
   M_stell=10.0 ** valid_sample_parameters['logmass_stell']
   Mdisk=10.0 ** valid_sample_parameters['logmass']
   R_in=valid_sample_parameters['Rin']
   R_out=5.0 * valid_sample_parameters['Rc']
   T_atm=valid_sample_parameters['tatm']
   pp=valid_sample_parameters['pp']
   Rc=valid_sample_parameters['Rc']
   incl=valid_sample_parameters['incl']
   vsys=valid_sample_parameters['vsys']
   pa=valid_sample_parameters['pa'] 
   xoff=valid_sample_parameters['xoff']
   yoff=valid_sample_parameters['yoff']
 
   T_mid=T_atm
   qq=-0.5
   v_turb=0.01
   X_co=1e-4
   flipme=True
   hanning=True
   
   unique_id=str(np.random.randint(1e10))
   model_name=f'{MCMC_OUTPUTS}/model_' + unique_id
   basics['distance']=distance

   # create a disk model using the sampled parameters
   x=Disk(
            q=qq, 
            McoG=Mdisk, 
            pp=pp, 
            Rin=R_in, 
            Rout=R_out, 
            Rc=Rc, 
            incl=incl,
            Mstar=M_stell, 
            Xco=X_co, 
            vturb=v_turb, 
            Zq0=0, 
            Tmid0=T_mid, 
            Tatm0=T_atm, 
            sigbound=[1e-3, float('inf')], 
            Rabund=[1,1000], 
            handed=-1
        )  

   # raytrace the disk model 
   total_model(
            x, 
            nchans=basics['nchans'] * 2 + 1, 
            chanstep=basics['chanstep'] / 2, 
            imres=basics['cell'],
            distance=basics['distance'],
            freq0=basics['freq'],
            vsys=vsys,
            obsv=basics['obsv'],
            chanmin=basics['chanmin'], 
            xnpix=basics['imsize'],
            PA=pa,
            modfile=model_name,
            flipme=flipme,
            hanning=hanning,
            Jnum=basics['Jnum'],
            offs=[xoff,yoff]
        )
    
   c=[]
   c.append(chiSq(basics["datafile"], model_name, dxy=basics['cell']/206265.0, dRA=0, dDec=0, pbcor=pbcor))

   # In case we have observations from more than one date: 
   for DATAFILE_i, _ in ADDITIONAL_OBS.items():   
      print(f"COMPUTING CHI SQR for: {DATAFILE_i}") 
      c.append(chiSq(DATAFILE_i, model_name, dxy=basics['cell']/206265.0, dRA=0, dDec=0, pbcor=pbcor))   
   os.remove(f'{model_name}.fits')
	
   return np.sum(c) * -0.5

def MCMC(
        init_params,
        params_uncertainty,
        param_order, 
        nsteps, 
        ndim, 
        nwalkers, 
        restart):

    '''
    reads in initial walker positions, starts MCMC chains, and saves results
    '''
    res,msg=valid_parameter_names(init_params, params_uncertainty, param_priors)
    if res:   
       param_locs=[init_params[param_name] for param_name in param_order]
       param_scales=[params_uncertainty[param_name] for param_name in param_order]
    else:
       print(f"Missing or Invalid parameter naming convention: {msg}")
       return 
    
    if restart==False:
        p0=np.random.normal(
            loc=param_locs,
            size=(nwalkers, ndim), 
            scale=param_scales)
        open(f"{MCMC_OUTPUTS}/{PARALLAX_FILE}", "w").close()  # clears the log file at the start
        # make the df header and create csv file 
        df=pd.DataFrame(columns=param_order+['lnprobs'])
        df.to_csv(f"{MCMC_OUTPUTS}/{CSV_FILE}", mode='w', header=True, index=True)
    else:
        print(f"WARNING: RESTARTING CHAIN USING {MCMC_OUTPUTS}/{CSV_FILE}")
        df=pd.read_csv(f"{MCMC_OUTPUTS}/{CSV_FILE}")
        p0=np.zeros([nwalkers,ndim])
        for i in range(nwalkers):
            for j, param_name in enumerate(param_order):
                p0[i,j]=df[param_name].iloc[-(nwalkers-i+1)]
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    init_len_df=len(df)
    print(f"STARTING FROM STEP {init_len_df//nwalkers}")

    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    run=sampler.sample(p0, iterations=nsteps, store=True, skip_initial_state_check=restart)
    
    for i, result in enumerate(run):
        pos, lnprobs, blob = result
        print(lnprobs)
        
        for k in range(nwalkers):
            new_row_values=np.append(pos[k], lnprobs[k])
            new_row_df=pd.DataFrame([dict(zip(param_order+['lnprob'],new_row_values))])
            new_row_df.to_csv(f"{MCMC_OUTPUTS}/{CSV_FILE}", mode='a', header=False, index=True)

        sys.stdout.write(f'completed step {(init_len_df+i)//nwalkers} out of {nsteps}\r')
        sys.stdout.flush()

    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:3f}".format(np.mean(sampler.acceptance_fraction)))
    return

pool=MPIPool()
start=time.time()
basics=calBase()
pbcor=makeGaussian(peak=1, size=basics['imsize'], FWHM=basics["FWHM"], center=None, PA=0)
PARALLAX_FILE='parallax_log.txt'
MCMC(   
        init_params,
        params_uncertainty,
        param_order, 
        restart=mcmc_hyper_params['restart'],
        nsteps=mcmc_hyper_params['nsteps'],
        ndim=mcmc_hyper_params['ndim'],
        nwalkers=mcmc_hyper_params['nwalkers']
)

print('Elapsed time (hrs):' , (time.time() - start)/3600)
