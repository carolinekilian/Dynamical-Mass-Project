from galario.double import sampleImage
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import emcee
import os
import shutil
import sys

#import the debris disk model

from disk_ecc import *
from raytrace import *

import time
start=time.time()

from schwimmbad import MPIPool
pool = MPIPool()


def calBase():
    datafile = "HD32297Jun21.uvfits"
    imfile = "Jun21COLineImage.fits"
    vsys = 5.55
    distance = 132.

    data_vis = fits.open(datafile)
    antenna = np.min(data_vis[2].data['diameter'])
    delta = data_vis[0].header['CDELT4']

    freq0 = data_vis[0].header['CRVAL4']
    if np.absolute(freq0 - 230.5e9) < np.absolute(freq0 - 345.7e9):
        freq = 230.53800000
        Jnum = 1
    else:
        freq = 345.79598990
        Jnum = 2
    
    basics = {"freq" : freq, "Jnum" : Jnum}
    n_spw = data_vis[0].header['NAXIS4']
    freq_end = freq0 + (n_spw - 1.0) / 2.0 * delta
    max_baseln = np.sqrt(np.max(data_vis[0].data['UU'] ** 2.0 + data_vis[0].data['VV'] ** 2.0)) * freq_end
    data_vis.close()

    im_vis = fits.open(imfile)

    vel0 = im_vis[0].header['CRVAL3']/1e3
    chanstep = im_vis[0].header['CDELT3']/1e3
    nstep = im_vis[0].header['NAXIS3']
    vel_fin = vel0 + nstep * chanstep
    obsv = np.arange(vel0, vel_fin, chanstep)

    chanstep = np.absolute(chanstep)
    nchans = int(2*np.ceil(np.abs(obsv-vsys).max()/chanstep)+1)
    chanmin = -(nchans/2.-.5)*chanstep
    im_vis.close()

    basics["obsv"] = obsv
    basics["chanstep"] = chanstep
    basics["nchans"] = nchans
    basics["chanmin"] = chanmin

    basics["vsys"] = vsys
    basics["distance"] = distance

    wavelength = 3e8/(basics["freq"] * 1.0e9)
    basics["cell"] = np.round(1.0/max_baseln * 20626.5, decimals=4)
    prim_bm = wavelength/antenna * 206265.0
    basics["imsize"] = int(2.0 ** np.round(np.log2((prim_bm) / basics["cell"])))
    basics["FWHM"] = np.round(1.13 * prim_bm/basics["cell"], decimals=2)
    basics["datafile"] = datafile

    return basics

#define equation to calculate chi-square

def makeGaussian(peak, size, FWHM=28, center=None, PA = 16): 

# params: 
# peak should be in units of Jy/pix
# size should be the length in pixels of one side of the array
# FWHMs should be in units of pixels
# center should be set to none if your gaussian is in center; otherwise written as (x,y)
# position angle should be in degrees
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    sigma = FWHM / 2.355
    
    PA = 90 + PA 
    PA = np.pi / 180. * PA  #convert to radians

    a = np.cos(PA)**2 / (2 * sigma**2) + np.sin(PA)**2 / (2 * sigma**2)
    b = -np.sin(2*PA) / (4 * sigma**2) + np.sin(2*PA) / (4 * sigma**2)
    b = -b
    c = np.sin(PA)**2 / (2 * sigma**2) + np.cos(PA)**2 / (2 * sigma**2)

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = size // 2 - center[0]
        y0 = size // 2 - center[1]

    return peak * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

val = calBase()


pbcor = makeGaussian(peak=1, size=val['imsize'], FWHM=val["FWHM"], center=None, PA=0)

# For additional primary beam corrections
# pbcor2 = makeGaussian(peak=1, size=600, FWHM=403.4, center=None, PA=0)
# pbcor3 = makeGaussian(peak=1, size=600, FWHM=403.4, center=None, PA=0)

#subtracted the outfile part
def chiSq(datafile, modfile, dxy, dRA, dDec, pbcor):
	
    mod_arr = []

    modfile_name = modfile + str('.fits')

    model_fits = fits.open(modfile_name)
    model = model_fits[0].data
    model = model.byteswap().newbyteorder().squeeze()
    model = model[:,::-1,:]
    model = model.copy(order='C')
   
    model_cor = model * pbcor
    model_cor = model_cor.copy(order='C')
    model_fits.close()

    chi = 0 # for calculation of chi-sq
    data_vis = fits.open(datafile)
    data_shape = data_vis[0].data['data'].shape
    delta = data_vis[0].header['CDELT4']
    freq0 = data_vis[0].header['CRVAL4'] 
    n_spw = data_vis[0].header['NAXIS4'] # number of spectral windows
    
    model_vis = np.zeros(data_shape)
    data = data_vis[0].data['data']

    freq_start = freq0 - (n_spw - 1.0) / 2.0 * delta

    for i in range(n_spw):
        freq = freq_start + i * delta
        u, v = (data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64)
    	
        foo = model_cor
        foo = np.require(foo, requirements='C')
        
        vis = sampleImage(foo[i,:,:], dxy/206265.0, u, v, dRA=dRA, dDec=dDec) 
        
        model_vis[:,0,0,0,i,0,0] = vis.real
        model_vis[:,0,0,0,i,1,0] = vis.real
        model_vis[:,0,0,0,i,0,1] = vis.imag
        model_vis[:,0,0,0,i,1,1] = vis.imag

        chi += ((vis.real - data[:,0,0,0,i,0,0])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.imag - data[:,0,0,0,i,0,1])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.real - data[:,0,0,0,i,1,0])**2 * data[:,0,0,0,i,1,2]).sum() + ((vis.imag - data[:,0,0,0,i,1,1])**2 * data[:,0,0,0,i,1,2]).sum()

    
    data_vis.close()
    

#    model_vis[:,0,0,0,:,0,2] = data[:,0,0,0,:,0,2] # weights (XX)
#    model_vis[:,0,0,0,:,1,2] = data[:,0,0,0,:,1,2] # weights (YY)
        
    #calculate chi-squared
    #nonzero = len(np.nonzero(data[:,0,0,0,0,0,2])[0])*2 +len(np.nonzero(data[:,0,0,0,0,1,2])[0])*2
    #chi2 += chi/nonzero
    
    return chi 

#define priors

def lnprob(p0):
    
    #new parameters
    logmass_stell, incl, Rc, qq, T_atm, pa, logmass, xoff, yoff, ecc, aop, vsys = p0 
    priors_logmass_stell = [-2, 2]
    priors_incl = [0, 90]
    priors_Rc = [0, 1000]
    priors_qq = [-5, 5] 
    priors_Tatm = [0, 500]
    priors_pa = [0, 360]
    priors_logmass = [-6, -2]
    priors_xoff = [-1.5, 1.5]
    priors_yoff = [-1.5, 1.5]
    priors_ecc = [0, 1]
    priors_aop = [0, 360]
    priors_vsys = [5, 6]


    if logmass_stell < priors_logmass_stell[0] or logmass_stell > priors_logmass_stell[1]:
        print("stellar mass out of bounds")
        return -np.inf

    if incl < priors_incl[0]or incl > priors_incl[1]:
        print("inclination angle out of bounds")
        return -np.inf

    if Rc < priors_Rc[0] or Rc > priors_Rc[1]:
        print("R_c out of bounds")
        return -np.inf

    if qq < priors_qq[0] or qq > priors_qq[1]:
        print("qq out of bounds")
        return -np.inf

    if T_atm < priors_Tatm[0] or T_atm > priors_Tatm[1]:
        print("t_atm (atmospheric temperature) out of bounds")
        return -np.inf

    if pa < priors_pa[0] or pa > priors_pa[1]:
        print("position angle out of bounds")
        return -np.inf

    if logmass < priors_logmass[0] or logmass > priors_logmass[1]:
        print("disk mass out of bounds")
        return -np.inf

    if xoff < priors_xoff[0] or xoff > priors_xoff[1]:
        print("x offset out of bounds")
        return -np.inf

    if yoff < priors_yoff[0] or yoff > priors_yoff[1]:
        print("y offset out of bounds")
        return -np.inf

    if ecc < priors_ecc[0] or ecc > priors_ecc[1]:
        print("eccentricity out of bounds")
        return -np.inf

    if aop < priors_aop[0] or aop > priors_aop[1]:
        print("angle of periastron out of bounds")
        return -np.inf

    if vsys < priors_vsys[0] or vsys > priors_vsys[1]:
        print("systemic velocity out of bounds")
        return -np.inf

    M_stell = 10.0 ** logmass_stell
    Mdisk = 10.0 ** logmass
    R_in = 0.1
    R_out = 5.0 * Rc
    T_mid = T_atm
    pp = -0.5
    v_turb = 0.01
    X_co = 1e-4
    flipme = True
    hanning = True

    x = Disk(q = qq, McoG = Mdisk, pp=pp,Ain=R_in, Aout=R_out, Rc=Rc, incl=incl, Mstar=M_stell, Xco=X_co, vturb=v_turb, Zq0=0, Tmid0=T_mid, Tatm0=T_atm, handed=-1, ecc=ecc, aop=aop, sigbound=[0, math.inf], Rabund=[1,1000])

    unique_id = str(np.random.randint(1e10))
    model_name = 'model_' + unique_id
    #model_file_name = model_name + '.model'

    #current editing point

    total_model(x, nchans = val['nchans'] * 2 + 1, chanstep=val['chanstep'] / 2, imres=val['cell'], distance=val['distance'],freq0=val['freq'], vsys = vsys, obsv = val['obsv'], chanmin = val['chanmin'], xnpix=val['imsize'], PA=pa, offs=[xoff, yoff], modfile=model_name, flipme=flipme, hanning=hanning, Jnum = val['Jnum'])
    c = []

    #make_model_vis(val["datafile"], model_name, isgas=True, freq0=val['freq'])
    c.append(chiSq(val["datafile"], model_name, dxy=val['cell'], dRA=0, dDec=0, pbcor=pbcor))

    #In case we have observations from more than one date

    #make_model_vis('m11dd.fits', model_name, isgas=False, freq0=freq0)
    #c.append(chiSq('m11dd.fits', model_name, dxy=1.1150715e-7, dRA=0, dDec=0, pbcor=pbcor))   
    #make_model_vis('m12dd.fits', model_name, isgas=False, freq0=freq0)
    #c.append(chiSq('m12dd.fits', model_name, dxy=1.1150715e-7, dRA=0, dDec=0, pbcor=pbcor))
    os.remove(model_name+'.fits')
	
    return np.sum(c) * -0.5

def MCMC(nsteps=4000, ndim=12, nwalkers=24, param_1=0.37, param_2=80.54, param_3=105.02, param_4= -0.37, param_5=21.3, param_6=46.6, param_7=-2.70, param_8=0.09, param_9=0.38, param_10 = 0.3, param_11 = 180.0, param_12 = 5.55, sigma_1=0.1, sigma_2=3, sigma_3=10, sigma_4=0.1, sigma_5=5, sigma_6=2, sigma_7=0.1, sigma_8=0.1, sigma_9=0.1, sigma_10 = 0.1, sigma_11 = 60, sigma_12 = 0.1, restart=False):

    '''Perform MCMC Affine invariants
    :param nsteps:       number of iterations
    :param ndim:         number of dimensions
    :param nwalkers:     number of walkers
    :param_1:     log of stellar mass
    :param_2:     inclination
    :param_3:     Critical radius
    :param_4:     Temperature radial power law index
    :param_5:     Atmospheric temperature
    :param_6:     position angle
    :param_7:     log of disk mass
    :param_8:     x positional offset
    :param_9:     y positional offset

    for any additional parameters to later add
    :param_8:          
    '''

#param

    if restart == False:

        p0 = np.random.normal(loc=(param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10, param_11, param_12), size=(nwalkers, ndim), scale=(sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7, sigma_8, sigma_9, sigma_10, sigma_11, sigma_12))

    else:
        a = pd.read_csv('HD32297_Nov2022_MCMC_Kazul.csv')
        p0 = np.zeros([nwalkers,ndim])
        for i in range(nwalkers):
            p0[i,0] = a['logmass_stell'].iloc[-(nwalkers-i+1)]
            p0[i,1] = a['incl'].iloc[-(nwalkers-i+1)]
            p0[i,2] = a['Rc'].iloc[-(nwalkers-i+1)]
            p0[i,3] = a['qq'].iloc[-(nwalkers-i+1)]
            p0[i,4] = a['T_atm'].iloc[-(nwalkers-i+1)]
            p0[i,5] = a['pa'].iloc[-(nwalkers-i+1)]
            p0[i,6] = a['logmass'].iloc[-(nwalkers-i+1)]
            p0[i,7] = a['xoff'].iloc[-(nwalkers-i+1)]
            p0[i,8] = a['yoff'].iloc[-(nwalkers-i+1)]
            p0[i,9] = a['ecc'].iloc[-(nwalkers-i+1)]
            p0[i,10] = a['aop'].iloc[-(nwalkers-i+1)]
            p0[i,11] = a['vsys'].iloc[-(nwalkers-i+1)]

            '''
            additional parameter space
            p0[i,7] = a['flux_6_aug'].iloc[-(nwalkers-i+1)]
            p0[i,8] = a['flux_6_mar'].iloc[-(nwalkers-i+1)]
            p0[i,9] = a['flux_6_jun'].iloc[-(nwalkers-i+1)]
            p0[i,10] = a['flux_9'].iloc[-(nwalkers-i+1)]
            p0[i,11] = a['h_6'].iloc[-(nwalkers-i+1)]
            p0[i,12] = a['h_ratio'].iloc[-(nwalkers-i+1)]
            p0[i,13] = a['beta'].iloc[-(nwalkers-i+1)]
            '''
        #read from csv here
 
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool) 
    run = sampler.sample(p0, iterations=nsteps, store=True)
    steps=[]

    for i, result in enumerate(run):
        pos, lnprobs, blob = result

        new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
        steps += new_step
        print(lnprobs)
        df = pd.DataFrame(steps)
        df.columns = ["logmass_stell", "incl", "Rc", "qq", "T_atm", "pa", "logmass", "xoff", "yoff", "ecc", "aop", "vsys", "lnprobs"]
        df.to_csv('HD32297_Nov1_2024_MCMC.csv', mode='w')
        sys.stdout.write('completed step {} out of {} \r'.format(i, nsteps) )
        sys.stdout.flush()
	
    #print(np.shape(sampler.chain))
    
    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:3f}".format(np.mean(sampler.acceptance_fraction)))

MCMC()

print('Elapsed time (hrs):' , (time.time() - start)/3600)


