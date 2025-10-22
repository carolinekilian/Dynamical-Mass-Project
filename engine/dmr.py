Author='David Vizghan'
import numpy as np 
import pandas as pd 
from matplotlib import cm 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
from matplotlib import rcParams 
from galario.double import sampleImage 
from astropy.io import fits 
import os 
import sys 
import subprocess  
from disk import * 
from raytrace import * 
from p0 import * 

qq=-0.5 							
v_turb=0.01 								
X_co=1e-4 
flipme=True
hanning=True
Zq0=0

def get_best_fit_params(csv_file=f"{MCMC_OUTPUTS}/{CSV_FILE}"): 
    """ TODO: extracts MCMC best parameters """
    df=pd.read_csv(csv_file) 
    burnin=10 
    nwalkers=mcmc_hyper_params['nwalkers']
    bf_params=dict()

    # best fit params are found in row where lnprob is maximized
    df=df.drop(df.index[0:burnin * nwalkers])
    index=df.lnprobs.idxmax()

    # load best fit params
    bf_params['logmass_stell']=df.logmass_stell[index] 				# stellar log mass in solar masses
    bf_params['incl']=df.incl[index] 						# inclination in degrees
    bf_params['Rc']=df.Rc[index] 						# characteristic radius - proxy for outer radius of the disk in au
    bf_params['pp']=df.pp[index] 						# gamma, disk CO surface density exponent 
    bf_params['pa']=df.pa[index] 						# position angle in degrees *presumably*
    bf_params['logmass']=df.logmass[index] 					# total CO mass, log mass,
    bf_params['Rin']=df.Rin[index] 						# inner boundary of disk density/temp calculation in au
    bf_params['xoff']=df.xoff[index] 						# x offset
    bf_params['yoff']=df.yoff[index] 						# y offset
    bf_params['vsys']=df.vsys[index]                                            # systemic velocity in km/s
    bf_params['tatm']=df.tatm[index] 						# Atmosphere temperature normalization, in K, defined at r=150 au
    return bf_params, df 

def calBase(datafile=DATAFILE, imfile=IMFILE, distance=DISTANCE_PC, vsys=init_params['vsys']):
    """
    extracts data from casa generated uvfits files and computes
    relevant observation parameters; used to construct disk model 
    
    ex args: 

    datafile="hales_cvel_1d.uvfits" 
    imfile="hales_cvel_1d.fits" 				
    distance=130.1 
    """
    
    # read out from visibility header
    data_vis=fits.open(datafile)
   
    delta=data_vis[0].header['CDELT4']
    n_spw=data_vis[0].header['NAXIS4']
    antenna=np.min(data_vis[2].data['diameter'])
    freq0=data_vis[0].header['CRVAL4']

    # read out from image header
    im_vis=fits.open(imfile)

    chanstep=im_vis[0].header['CDELT3']/1e3
    nstep=im_vis[0].header['NAXIS3']
    vel0=im_vis[0].header['CRVAL3']/1e3
 
    im_vis.close()
    
    # define the basics dictionary 
    if np.absolute(freq0 - 230.5e9) < np.absolute(freq0 - 345.7e9):
        freq=230.53800000
        Jnum=1
    else:
        freq=345.79598990
        Jnum=2
    basics={"freq" : freq, "Jnum" : Jnum}
    
    wavelength=3e8/(basics["freq"] * 1.0e9)  
    prim_bm=wavelength/antenna * 206265.0
    freq_end=freq0 + (n_spw - 1.0) / 2.0 * delta
    max_baseln=np.sqrt(np.max(data_vis[0].data['UU'] ** 2.0 + data_vis[0].data['VV'] ** 2.0)) * freq_end    
    vel_fin=vel0 + nstep * chanstep
    obsv=np.arange(vel0, vel_fin, chanstep)
    chanstep=np.absolute(chanstep)
    nchans=int(2*np.ceil(np.abs(obsv-vsys).max()/chanstep)+1)
    chanmin=-(nchans/2.-.5)*chanstep 

    basics["obsv"]=obsv
    basics["chanstep"]=chanstep
    basics["nchans"]=nchans
    basics["chanmin"]=chanmin
    basics["vsys"]=vsys
    basics["distance"]=distance
    basics["cell"]=np.round(1.0/max_baseln * 20626.5, decimals=4)
    basics["imsize"]=int(2.0 ** np.round(np.log2((prim_bm) / basics["cell"])))
    basics["FWHM"]=np.round(1.13 * prim_bm/basics["cell"], decimals=2)
    basics["datafile"]=datafile
    
    data_vis.close()
    im_vis.close()
    return basics

def makeGaussian(peak, size, FWHM=28, center=None, PA=16): 
    """
    The ALMA primary beam can be approximated to be a 2D Gaussian with a full-width 
    half-maximum (FWHM) of 1.13 L/D, with L being the wavelength of observation and 
    D being the antenna diameter. 

    - peak [Jy/pixels]
    - imsize [int]: image size in pixels
    - FWHMs [pixels]
    - center: set to none if your gaussian is in center; otherwise written as (x,y)
    - position angle [degrees]
    """
    
    x=np.arange(0, size, 1, float)
    y=x[:,np.newaxis]

    sigma=FWHM / 2.355    
    PA=np.pi / 180. * (PA+90)  #convert to radians

    a=np.cos(PA)**2 / (2 * sigma**2) + np.sin(PA)**2 / (2 * sigma**2)
    b=-np.sin(2*PA) / (4 * sigma**2) + np.sin(2*PA) / (4 * sigma**2)
    b=-b
    c=np.sin(PA)**2 / (2 * sigma**2) + np.cos(PA)**2 / (2 * sigma**2)

    if center is None:
        x0=y0=size // 2
    else:
        x0=size // 2 - center[0]
        y0=size // 2 - center[1]

    return peak * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def ChiSqr(datafile, modfile, fileout, dxy, dRA, dDec, residual, pbcor): 
    """
    applies primary beam correction to your modfile and computes chi-squared

    datafile [str]=your disk data file (include .fits)
    modfile [str]=your model image name; crucially 
    fileout [str]= name of your output file 
    dxy [int]=pixel size in arcsec (from the galario documentation)
    dRA [int/float]=Right ascension offset in radian (from the galario documentation)
    dDec [int/float]=declination offset in radian (from the galario documentation)
    residual [bool]=If True the function will subtract your model disk from the 
                      disk data WITHIN THE VISIBILITY DOMAIN! 

		      If Fakse, turn your model disk
                      into data visibilities by using the data structure of your data visibilities. For instance, if you
                      made a model of AU Microscopii's debris disk, your datafile is used regardless as a structure for
                      converting your model into a datafile which is compatible and identical in shape to the actual data.
    
    pbcor=primary beam correction array, which is additionally multiplied onto the model to account for the
            effects of ALMA's primary beam on the final image.
    """ 
    modfile_name=modfile + str('.fits')                                               
    model_fits=fits.open(modfile_name) 
    model=model_fits[0].data	     
    model=model.byteswap().newbyteorder().squeeze() 
    model=model[:,::-1,:] # flips the data on one axis
    model=model.copy(order='C') 
    model_cor=model * pbcor # the crucial step -- multiplying the model by your primary beam correction in 
                            # the image domain before converting the model IMAGE into model VISIBILITIES. 
    model_cor=model_cor.copy(order='C') 
    model_fits.close()  

    data_vis=fits.open(datafile) 
    data_shape=data_vis[0].data['data'].shape 
    delta_freq=data_vis[0].header['CDELT4'] # channel width
    freq0=data_vis[0].header['CRVAL4']      # central frequency of observation 
    n_spw=data_vis[0].header['NAXIS4']      # number of spectral windows 
    model_vis=np.zeros(data_shape) 
    data=data_vis[0].data['data'] 
	
    chi=0
    freq_start=freq0 - (n_spw - 1.0) / 2.0 * delta_freq

    for i in range(n_spw):
        if i in EXCLUDED_CHANNEL_IDXs:
           print(f"EXCLUDING CHANNEL {i} FROM CHI SQR")
           continue 

        freq=freq_start + i * delta_freq
        u,v=(data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64) 
        model_cor=np.require(model_cor, requirements='C')
        # Galario: takes model image and creates corresponding visibilities
        vis=sampleImage(model_cor[i,:,:], dxy, u, v, dRA=dRA, dDec=dDec)
                
        # look at the documentation for the sampleImage function, but to recap...
        # model_cor is the model image
        # dxy is the pixel size
        # u, v are the U, V tracks of the data 
        # dRA and dDec are offsets in RA and declination
         	
        # throw model visibilities into correspondin
        # sections of the model array "skeleton" (XX and YY polarisations)
        model_vis[:,0,0,0,i,0,0]=vis.real
        model_vis[:,0,0,0,i,1,0]=vis.real
        model_vis[:,0,0,0,i,0,1]=vis.imag
        model_vis[:,0,0,0,i,1,1]=vis.imag 
        chi +=((vis.real - data[:,0,0,0,i,0,0])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.imag - data[:,0,0,0,i,0,1])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.real - data[:,0,0,0,i,1,0])**2 * data[:,0,0,0,i,1,2]).sum() + ((vis.imag - data[:,0,0,0,i,1,1])**2 * data[:,0,0,0,i,1,2]).sum()
        # note -- the shape [:,0,0,0,0,0,0] etc. may not be the same as the shape of your data. 

    if residual==False:
       model_vis[:,0,0,0,:,0,2]=data[:,0,0,0,:,0,2]                                                     
       model_vis[:,0,0,0,:,1,2]=data[:,0,0,0,:,1,2]     
       data_vis[0].data['data']=model_vis
       data_vis.writeto(fileout, overwrite=True) 
       data_vis.close()
        
    if residual==True:
       data_vis[0].data['data']=data_vis[0].data['data'] - model_vis
       data_vis.writeto(fileout, overwrite=True)
       data_vis.close() 
    return 

def create_dmr():
    """
    create data, model, residual images by creating disk model, raytracing it, and then generating visibilities using galario 

    intended use case: 
    1. when performing chi-by-eye (i.e choosing your initialization parameters for MCMC by inspecting match between 
    data and model images)
    2. viewing disk model using mcmc best fit parameters
    """

    basics=calBase()
   
    use_case=input("p0 or MCMC: ")
    if use_case=='p0':	
       P0_MODEL_NAME=f'{DMR_PREFIX}_PYTHON_P0_ModelDisk'
       print("USING PARAMETERS IN p0.py")
       print(f"init_params: {init_params}")
       MODEL_NAME=P0_MODEL_NAME
       x=Disk(
         q=qq,
         McoG=10**init_params['logmass'],
         pp=init_params['pp'],
         Rin=init_params['Rin'],
         Mstar=10**init_params['logmass_stell'],
         Xco=X_co,
         vturb=v_turb,
         Zq0=Zq0,
         Tmid0=init_params['tatm'],
         Tatm0=init_params['tatm'],
         sigbound=[1e-3, float('inf')],
         Rabund=[1,1000],
         incl=init_params['incl'],
         Rout=5*init_params['Rc'],
         Rc=init_params['Rc']
         )

       total_model(
         x, 
         nchans=basics['nchans'] * 2 + 1,
         chanstep=basics['chanstep'] / 2, 
         imres=basics['cell'],
         distance=basics['distance'],
         freq0=basics['freq'],
         vsys=basics['vsys'], 
         obsv=basics['obsv'], 
         chanmin=basics['chanmin'],
         xnpix=basics['imsize'], 
         PA=init_params['pa'],
         modfile=MODEL_NAME,
         flipme=flipme,
         bin=3,
         hanning=hanning, 
         Jnum=basics['Jnum'],
         offs=[init_params['xoff'],init_params['yoff']]
         )

    elif use_case=='MCMC': 
       MCMC_MODEL_NAME=f'{DMR_PREFIX}_PYTHON_MCMC_BFP_ModelDisk'
       print(f"USING PARAMTERS IN CSV FILE: {MCMC_OUTPUTS}/{CSV_FILE}")
       bf_params,_=get_best_fit_params() 
       print(f"bf_params: {bf_params}") 
       MODEL_NAME=MCMC_MODEL_NAME
       x=Disk(
         q=qq,
         McoG=10**bf_params['logmass'],
         pp=bf_params['pp'],
         Rin=bf_params['Rin'],
         Mstar=10**bf_params['logmass_stell'],
         Xco=X_co,
         vturb=v_turb,
         Zq0=Zq0,
         Tmid0=bf_params['tatm'],
         Tatm0=bf_params['tatm'],
         sigbound=[1e-3, float('inf')],
         Rabund=[1,1000],
         incl=bf_params['incl'],
         Rout=5*bf_params['Rc'],
         Rc=bf_params['Rc']
         )

       total_model(
         x, 
         nchans=basics['nchans'] * 2 + 1,
         chanstep=basics['chanstep'] / 2, 
         imres=basics['cell'],
         distance=basics['distance'],
         freq0=basics['freq'],
         vsys=basics['vsys'], 
         obsv=basics['obsv'], 
         chanmin=basics['chanmin'],
         xnpix=basics['imsize'], 
         PA=bf_params['pa'],
         modfile=MODEL_NAME,
         flipme=flipme,
         bin=3,
         hanning=hanning, 
         Jnum=basics['Jnum'],
         offs=[bf_params['xoff'],bf_params['yoff']]
         )
    else: 
       print("Not a valid input.")
    # create idealized primary beam 
    pbcor1=makeGaussian(peak=1, size=basics['imsize'], FWHM=basics["FWHM"], center=None, PA=0)

    # This section creates the residual images
    ChiSqr(
           datafile=basics['datafile'],
           modfile=MODEL_NAME,
           fileout=INPUT_RESID_DMR,
           dxy=basics['cell']/206265., 
           dRA=0,
           dDec=0,
           residual=True,
           pbcor=pbcor1
          )

    # This section creates the best-fit images
    ChiSqr(
           datafile=basics['datafile'], 
           modfile=MODEL_NAME, 
           fileout=INPUT_MODEL_DMR, 
           dxy=basics['cell']/206265., 
           dRA=0, 
           dDec=0, 
           residual=False,
           pbcor=pbcor1
          )

    # I use a shell script here for this which uses standard miriad commands (invert etc.)
    subprocess.call(['./dmr-script.sh'])
    
    # this actually just triggers all of the code in specfig
    import specfig as spec
    spec.make_spectrum_plot(label=use_case)
    return 

if __name__ == "__main__":
   run=input("Run dmr? (yes or no): ")
   if run=='yes':
      create_dmr()
   bf_params, _ =get_best_fit_params()
   print(f"BESTFIT PARAMETERS: {bf_params}")
    
