import numpy as np 
from matplotlib import cm 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
from matplotlib import rcParams 
from astropy.io import fits 

rcParams['font.family'] = 'serif'
#image = fits.open('AUMic_band9.fits') # good! 
#model = fits.open('HD32297Jan1_model_map.fits')
#resid = fits.open('HD32297Jan1_resid_map.fits')

image=fits.open('/Volumes/disks/josh/HD155853/1748/HD155853_Aug30_dirty_robust=0.5.fits')
model=fits.open('HD155853_model.fits')
resid=fits.open('HD155853_resid.fits')

hdr=image[0].header

RA = hdr['cdelt1'] * (np.arange(hdr['naxis1'])-hdr['naxis1']/2. + 0.5) * 3600.0
dec = -1 * RA

cmap = cm.get_cmap('Reds')

# I print out the best-fit values of my parameters for my own sake
"""
print('After ' + str(len(df.M_disk)/50) + ' steps...')
print('...')
print('Disk mass (solar mass): ' + str(M_bestfit))
print('power law index: ' + str(p_bestfit))
print('inclination (deg): ' + str(incl_bestfit))
print('position angle (deg): ' + str(PA_bestfit))
print('stellar flux (Jy): ' + str(flux_bestfit))
print('inner radius (au): ' + str(R_in_bestfit))
print('outer radius (au): ' + str(R_out_bestfit))
print('scale height: ' + str(h_bestfit))
"""
# Now I take out my data from the image, model, residual for the sake of plotting

im = image[0].data.squeeze() * 1e3
im = np.flip(im, axis=0)
md = model[0].data.squeeze() * 1e3
md = np.flip(md, axis=0)
rs = resid[0].data.squeeze() * 1e3
rs = np.flip(rs, axis=0)

# plot

fig = plt.figure(figsize=(10,4))
fig.subplots_adjust(wspace=0)
gs = gridspec.GridSpec(1,4,width_ratios=[3,3,3,0.15])

ax1 = fig.add_subplot(gs[0])
img = ax1.imshow(im, extent=[max(RA), min(RA), min(RA), max(RA)], cmap=cmap, vmin=np.min(im), vmax=np.max(im))
ax1.contour(np.flip(im, axis=0), 0.000283e3 * np.array([2,4,6,8,10]), linewidths=0.8, colors='black', extent=[max(RA), min(RA), min(RA), max(RA)])
ax1.set_aspect('equal')

ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax2.imshow(md, extent=[max(RA), min(RA), min(RA), max(RA)], cmap=cmap, vmin=np.min(im), vmax=np.max(im))

ax2.contour(np.flip(md, axis=0), 0.000283e3 * np.array([2,4,6,8,10]), linewidths=0.8, colors='black', extent=[max(RA), min(RA), min(RA), max(RA)])
ax2.set_aspect('equal')

ax3 = fig.add_subplot(gs[2],sharey=ax1)
ax3.imshow(rs, extent=[max(RA), min(RA), min(RA), max(RA)], cmap=cmap, vmin=np.min(im), vmax=np.max(im))
ax3.contour(np.flip(rs, axis=0), 0.000283e3 * np.array([2,4,6,8,10]), linewidths=0.8, colors='black', extent=[max(RA), min(RA), min(RA), max(RA)])
ax3.set_aspect('equal')

ax1.set_xlim([3.95,-3.95])
ax1.set_ylim([-3.95,3.95])

ax2.set_xlim([3.95,-3.95])
#ax2.tick_params(labeltop=False)

ax3.set_xlim([3.95,-3.95])
#ax3.tick_params(labeltop=False)

ax1.set_ylabel(ylabel=r'$\Delta\delta$ (")')
ax1.set_xlabel(xlabel=r'$\Delta\alpha$ (")')
ax2.set_xlabel(xlabel=r'$\Delta\alpha$ (")')
ax3.set_xlabel(xlabel=r'$\Delta\alpha$ (")')

ax1.annotate('AU Mic', (3.7, 3.5), color='black', fontsize=11)
ax1.annotate(r'ALMA 450 $\mu$m', (3.7, 3), color='black', fontsize=11)
ax2.annotate('Model', (3.7, 3.5), color='black', fontsize=11)
ax3.annotate('Residual', (3.7, 3.5), color='black', fontsize=11)


ax4 = fig.add_subplot(gs[3])
cbar = plt.colorbar(img, cax=ax4, orientation='vertical', pad=0.02)
cbar.set_label(r'mJy/bm')
ax4.set_aspect(20)

axes = [ax1, ax2, ax3]

for ax in axes:
	try:
		ax.label_outer()
	except:
		pass
line_x = np.linspace(2, 2.9903, 2)
line_y = np.ones( 2)

ax1.errorbar(-line_x, -2.96 * line_y, yerr=0.1, color='black')
ax1.annotate('10 au', (-1.81, -3.5), color='black', fontsize=11)

ax2.errorbar(-line_x, -2.96 * line_y, yerr=0.1, color='black')
