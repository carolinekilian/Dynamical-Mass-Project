from galario.double import sampleImage
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import emcee
import os
import shutil
import sys
import subprocess
import bettermoments as bm
#import the debris disk model

from disk import *
from raytrace import *

import time
start=time.time()

from schwimmbad import MPIPool
pool = MPIPool()
def calBase():
    datafile1 = "HD121617.LB.uvfits"
    datafile2 = "HD121617.SB.uvfits"
    imfile = "HD121617_12CO_30chans_robust_0.5_combined"
    vsys = 7.0
    distance = 117

    data_vis = fits.open(datafile1)
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

    im_vis = fits.open(imfile+'.fits')

    vel0 = im_vis[0].header['CRVAL3']/1e3
    chanstep = im_vis[0].header['CDELT3']/1e3
    nstep = im_vis[0].header['NAXIS3']
    vel_fin = vel0 + nstep * chanstep
    obsv = np.arange(vel0, vel_fin, chanstep)

    chanstep = np.absolute(chanstep)
    nchans = 2*np.ceil(np.abs(obsv-vsys).max()/np.abs(chanstep))+1
    chanmin = -(nchans/2.-.5)*chanstep
    nchans = int(nchans)
    im_vis.close()

    obsv = obsv - 0.062
    basics["obsv"] = obsv
    basics["chanstep"] = -chanstep
    basics["nchans"] = nchans
    basics["chanmin"] = -chanmin

    basics["vsys"] = vsys
    basics["distance"] = distance

    wavelength = 3e8/(basics["freq"] * 1.0e9)
    #basics["cell"] = np.round(1.0/max_baseln * 20626.5, decimals=4)
    basics["cell"] = 0.01
    prim_bm = wavelength/antenna * 206265.0
    #basics["imsize"] = int(2.0 ** np.round(np.log2((prim_bm) / basics["cell"])))
    basics["imsize"] = 512
    basics["FWHM"] = np.round(1.13 * prim_bm/basics["cell"], decimals=2)
    basics["datafile1"] = datafile1
    basics["datafile2"] = datafile2
    
    datamomentfile = fits.open(imfile+'_M9.fits')
    basics["datamoment"] = datamomentfile[0].data
    datamomentfile.close()
    dataerrorfile = fits.open(imfile+'_dM9.fits')
    basics["dataerror"] = dataerrorfile[0].data
    dataerrorfile.close()

    return basics

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

pbcor1 = makeGaussian(peak=1, size=val['imsize'], FWHM=val["FWHM"], center=None, PA=16)

# For additional primary beam corrections
# pbcor2 = makeGaussian(peak=1, size=600, FWHM=403.4, center=None, PA=0)
# pbcor3 = makeGaussian(peak=1, size=600, FWHM=403.4, center=None, PA=0)

def visibilities(datafile, modfile, fileout, dxy, dRA, dDec, pbcor, unique_id):
# see dmr_moment1 code for detailed comments
    modfile_name = modfile + str('.fits')
    model_fits = fits.open(modfile_name)
    model = model_fits[0].data
    model = model.byteswap().newbyteorder().squeeze()
    model = model[:,::-1,:]
    model = model.copy(order='C')
   
    model_cor = model * pbcor
    model_cor = model_cor.copy(order='C')
    model_fits.close()

    data_vis = fits.open(datafile)
    data_shape = data_vis[0].data['data'].shape
    delta_freq = data_vis[0].header['CDELT4']
    freq0 = data_vis[0].header['CRVAL4'] 
    n_spw = data_vis[0].header['NAXIS4'] # number of spectral windows
    
    model_vis = np.zeros(data_shape)
    data = data_vis[0].data['data']

    freq_start = freq0 - (n_spw - 1.0) / 2.0 * delta_freq

    for i in range(n_spw):
        freq = freq_start + i * delta_freq
        u, v = (data_vis[0].data['UU'] * freq).astype(np.float64), (data_vis[0].data['VV'] * freq).astype(np.float64)
    	
        foo = model_cor
        foo = np.require(foo, requirements='C')
        
        vis = sampleImage(foo[i,:,:], dxy, u, v, dRA=dRA, dDec=dDec) 
        
        model_vis[:,0,0,0,i,0,0] = vis.real
        model_vis[:,0,0,0,i,1,0] = vis.real
        model_vis[:,0,0,0,i,0,1] = vis.imag
        model_vis[:,0,0,0,i,1,1] = vis.imag

        # chi += ((vis.real - data[:,0,0,0,i,0,0])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.imag - data[:,0,0,0,i,0,1])**2 * data[:,0,0,0,i,0,2]).sum() + ((vis.real - data[:,0,0,0,i,1,0])**2 * data[:,0,0,0,i,1,2]).sum() + ((vis.imag - data[:,0,0,0,i,1,1])**2 * data[:,0,0,0,i,1,2]).sum()

    model_vis[:,0,0,0,:,0,2] = data[:,0,0,0,:,0,2]
    model_vis[:,0,0,0,:,1,2] = data[:,0,0,0,:,1,2]

    data_vis[0].data['data'] = model_vis
    outfilename=fileout+unique_id+".uvfits"
    data_vis.writeto(outfilename, overwrite=True)
    data_vis.close()

def chiSq(modelfile, velaxis, rms_clip, unique_id):
    visibilities(val['datafile1'], modelfile, fileout='model_DMR_LB', dxy=val['cell']/206265., dRA=0, dDec=0, pbcor=pbcor1, unique_id=unique_id)
    visibilities(val['datafile2'], modelfile, fileout='model_DMR_SB', dxy=val['cell']/206265., dRA=0, dDec=0, pbcor=pbcor1, unique_id=unique_id)

    # shell script to clean the model images
    subprocess.call(['./clean-model.sh', unique_id])

    # prepare to create moment9 map of cleaned model image
    data1, velaxis1 = bm.load_cube(modelfile+'.fits')
    data2, velaxis2 = bm.load_cube(velaxis)
    rms = bm.estimate_RMS(data=data1, N=10)
    
    # mask 1
    modelmask = bm.get_threshold_mask(data=data1, clip=rms_clip, rms=rms)
    masked_data = data1 * modelmask
    
    # collapse to moment 9
    moments = bm.collapse_ninth(velax=velaxis2, data=masked_data, rms=rms)
    mom_data, mom_error = moments
    loc = np.where(np.max(masked_data, axis=0)==0)
    mom_data[loc] = 0
    
    # load moment9 of data and prepare for computation
    data_data = val['datamoment']
    data_data[np.isnan(data_data)] = 0 #This line really only needed for mosaiced disks

    # load error from data
    data_error = val['dataerror']
    data_error[np.isnan(data_error)] = 0

    raw_chi = np.sum((data_data-mom_data) ** 2 / (mom_error **2 ))
    
    return raw_chi

#define priors
def lnprob(p0):
    
    parallax_mean = 8.4825e-3
    parallax_std = 0.0323e-3
    parallax = np.random.normal(parallax_mean, parallax_std)
    while parallax <= 0:
        parallax = np.random.normal(parallax_mean, parallax_std)
        print("parallax out of bounds")

    distance = 1 / parallax

    with open("parallax_log.txt", "a") as f:
        f.write(f"parallax: {parallax:.6e} arcsecs; distance = {distance:.2f} pc \n")

    #new parameters
    logmass_stell, Rc, pp, vsys, Rin, incl, xoff, yoff, pa = p0 
    priors_logmass_stell = [-1, 1]
    priors_incl = [20, 70]
    priors_Rc = [0, 200]
    priors_Rin = [0, 100]
    priors_pp = [-5, 5] #???
    priors_pa = [0, 360]
    priors_vsys = [3,15]
    priors_xoff = [-1, 1]
    priors_yoff = [-1, 1]
    
    if logmass_stell < priors_logmass_stell[0] or logmass_stell > priors_logmass_stell[1]:
        print("stellar mass out of bounds")
        return -np.inf

    if incl < priors_incl[0]or incl > priors_incl[1]:
        print("inclination angle out of bounds")
        return -np.inf

    if Rc < priors_Rc[0] or Rc > priors_Rc[1]:
        print("R_c out of bounds")
        return -np.inf

    if Rin < priors_Rin[0] or Rin > Rc or Rin > priors_Rin[1]:
        print("R_in out of bounds")
        return -np.inf

    if pp < priors_pp[0] or pp > priors_pp[1]:
        print("pp out of bounds")
        return -np.inf

    if vsys < priors_vsys[0] or vsys > priors_vsys[1]:
        print("vsys out of bounds")
        return -np.inf

    if xoff < priors_xoff[0] or xoff > priors_xoff[1]:
        print("xoff out of bounds")
        return -np.inf

    if yoff < priors_yoff[0] or yoff > priors_yoff[1]:
        print("yoff out of bounds")
        return -np.inf

    if pa < priors_pa[0] or pa > priors_pa[1]:
        print("pa out of bounds")
        return -np.inf

    M_stell = 10.0 ** logmass_stell
    Mdisk = 10.0 ** -7
    #xoff = 0.0
    #yoff = 0.0
    R_in = Rin
    R_out = 5.0 * Rc
    T_atm = 75.
    T_mid = T_atm
    # incl = 43.8
    qq = -0.5
    v_turb = 0.01
    X_co = 1e-4
    flipme = True
    hanning = True

    x = Disk(McoG=Mdisk, pp=pp, Rin=R_in, Rout=R_out, Rc=Rc, incl=incl, Mstar=M_stell, Zq0=80, Tmid0=T_mid, Tatm0=T_atm)
    unique_id = str(np.random.randint(1e10))
    model_name = 'model_' + unique_id
    #model_file_name = model_name + '.model'

    total_model(x, nchans = val['nchans'] * 2 + 1, manual_chan_params = True, chanstep=val['chanstep'] / 2, imres=val['cell'], distance=val['distance'],freq0=val['freq'], vsys = vsys, obsv = val['obsv'], chanmin = val['chanmin'], xnpix=val['imsize'], PA=pa, modfile=model_name, flipme=flipme, hanning=hanning, Jnum = val['Jnum'], offs=[xoff,yoff])

    chi = chiSq(model_name, 'HD121617_12CO_30chans_robust_0.5_combined.fits', 5.0, unique_id)
                                                                                                                                    #Doubled 
    #In case we have observations from more than one date

    #make_model_vis('m11dd.fits', model_name, isgas=False, freq0=freq0)
    #c.append(chiSq('m11dd.fits', model_name, dxy=1.1150715e-7, dRA=0, dDec=0, pbcor=pbcor))   
    #make_model_vis('m12dd.fits', model_name, isgas=False, freq0=freq0)
    #c.append(chiSq('m12dd.fits', model_name, dxy=1.1150715e-7, dRA=0, dDec=0, pbcor=pbcor))
    os.remove(model_name+'.fits')
    os.remove('modelmom1_'+unique_id+'.fits')
    return -0.5 * chi

def MCMC(nsteps=2600, ndim=9, nwalkers=40, param_1=0.28, param_2=80.4, param_3= -0.5, param_4 = 7.0, param_5 = 5, param_6 = 43.8, param_7=0.0, param_8=0.0, param_9=60.0, sigma_1=0.05, sigma_2=0.5, sigma_3=0.1, sigma_4 = 0.2, sigma_5 = 1, sigma_6 = 1, sigma_7=0.01, sigma_8=0.01, sigma_9=2, restart=False):
 # fix x and y, fix logmass
    '''Perform MCMC Affine invariants
    :param nsteps:       number of iterations
    :param ndim:         number of dimensions
    :param nwalkers:     number of walkers
    :param_1:     log of stellar mass
    :param_2:     inclination
    :param_3:     Critical radius
    :param_4:     Surface density radial power law index
    :param_5:     position angle
    :param_6:     log of disk mass

    for any additional parameters to later add
    :param_7:          
    '''

#param

    if restart == False:

        p0 = np.random.normal(loc=(param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9), size=(nwalkers, ndim), scale=(sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7, sigma_8, sigma_9))

    else:
        a = pd.read_csv('jun1_2023.csv')
        p0 = np.zeros([nwalkers,ndim])
        for i in range(nwalkers):
            p0[i,0] = a['logmass_stell'].iloc[-(nwalkers-i+1)]
            p0[i,1] = a['incl'].iloc[-(nwalkers-i+1)]
            p0[i,2] = a['Rc'].iloc[-(nwalkers-i+1)]
            p0[i,3] = a['pp'].iloc[-(nwalkers-i+1)]
            p0[i,4] = a['vsys'].iloc[-(nwalkers-i+1)]
            p0[i,5] = a['Rin'].iloc[-(nwalkers-i+1)]

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
        df.columns = ["logmass_stell", "Rc", "pp", "vsys", "Rin", "incl", "xoff", "yoff", "pa", "lnprobs"]
        df.to_csv('july24_2025.csv', mode='w')
        sys.stdout.write('completed step {} out of {} \r'.format(i, nsteps) )
        sys.stdout.flush()
	
    #print(np.shape(sampler.chain))
    
    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:3f}".format(np.mean(sampler.acceptance_fraction)))

MCMC()

print('Elapsed time (hrs):' , (time.time() - start)/3600)


