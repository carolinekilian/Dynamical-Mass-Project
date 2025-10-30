import bettermoments as bm
import subprocess
from astropy.io import fits
import numpy as np

data, velaxis = bm.load_cube('HD121617_12CO_30chans_robust_0.5_combined.fits')
rms = bm.estimate_RMS(data=data, N=5)
threshold_mask = bm.get_threshold_mask(data=data, clip=5.0)
masked_data = threshold_mask * data
datamoment = bm.collapse_ninth(velax=velaxis, data=masked_data, rms=rms)
datamoment_data, datamoment_error = datamoment
loc = np.where(np.max(masked_data, axis=0)== 0)
datamoment_data[loc] = np.nan
datamoment = datamoment_data, datamoment_error
bm.save_to_FITS(moments=datamoment, method='ninth', path='HD121617_12CO_30chans_robust_0.5_combined.fits')

mask_mom = bm.collapse_ninth(velax=velaxis, data=threshold_mask, rms=rms)
mask_mom_data, mask_mom_error = mask_mom
loc = np.where(np.max(threshold_mask, axis=0)==0)
mask_mom_data[loc] = np.nan
not_loc = np.where(np.max(threshold_mask, axis=0)!= 0)
mask_mom_data[not_loc] = 1
hdu = fits.PrimaryHDU(mask_mom_data)
hdu.writeto('thresholdmaskmom1.fits', overwrite=True)

