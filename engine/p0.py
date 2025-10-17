from astropy import units as u
import numpy as np
# names for subdirectories
CASA_DATA='CASA_DATA'
DMR_OUTPUTS='DMR_OUTPUTS'
MCMC_OUTPUTS='MCMC_OUTPUTS'
# prefixes for dmr-generated files 
DMR_PREFIX='DMR_RES'
INPUT_MODEL_DMR=f'{DMR_PREFIX}_PYTHON_model.fits'  
INPUT_RESID_DMR=f'{DMR_PREFIX}_PYTHON_resid.fits'

#  SOME REMINDERS
#  STOP! DID YOU RUN VAR VIS AND ADD WEIGHTS? 
#  STOP! DID YOU MAKE SURE TO RENAME YOUR FILES AFTER YOU RAN VAR VIS?
#  STOP! DID YOU CHANGE THE CSV FILE NAME?
#  STOP! DID YOU EXCLUDE ALL FLAGGED CHANNELS FROM YOUR CHI SQR?

#################################################

# ONLY edit VARIABLES below this line
# DO NOT try running dmr files with any additional print statements as dmr-script.sh read variables form stdout (i.e stuff printed in terminal)
# if you really need to add print statements do so in the 'playground' at the bottom of this file 
DISKNAME='HD152989'
IMFILE=f'{CASA_DATA}/HD152989_Aug23_18chan.fits'
DATAFILE=f'{CASA_DATA}/HD152989_Aug23_18chan_var_vis.uvfits'
CSV_FILE=f'{DISKNAME}_Oct_13_MCMC.csv'
#################################################

# MUST FOLLOW convention below: f'{CASA_DATA}/<filename>.uvfits': f'{CASA_DATA}/<filename>.fits',
# NOTE: ALL .uvfits and .fits FILES MUST BE ON THE SAME VELOCITY GRID (i.e they should have the same velocities -- up to the ten-thousandths --
#       for all channels) 
ADDITIONAL_OBS={
 #f'{CASA_DATA}/secondfile.uvfits' : f'{CASA_DATA}/secondfile.fits',
 # ...
}
#################################################

# Ideally this is the same pixel size as used in tclean -- this is used for GALARIO tasks
PIXEL_SIZE=0.024787369505505884

#################################################

# If you have contaminated channels input their indicies here (0-indexed) 
EXCLUDED_CHANNEL_IDXs=[] # ex: [4,5,8]
#################################################

# Use SIMBAD (to get these distance measurements)
# NOTE: I converted to arcsec, SIMBAD gives you measurements in milli-arcsec
# https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=HD152989&submit=SIMBAD+search
SIMBAD_parallax_arcsec=8.6755/1000
SIMBAD_parallax_uncertainty_arcsec=0.0317/1000
DISTANCE_PC=1/SIMBAD_parallax_arcsec

#################################################

# ONLY MODIFY THIS LIST BY ADDING PARAMETERS TO THE END 
# IF YOU MODIFY any of param_order, init_params, params_uncertainty, param_priors, or name_map
# THEN YOU MUST MODIFY ALL 
param_order=[
            "logmass_stell", 
            "incl", 
            "Rc", 
            "pp", 
            "pa", 
            "logmass",
            "Rin", 
            "xoff",
            "yoff",
            "vsys", 
            "tatm",
            #"new_parameter_name1",
            #...,
]

# the intitial parameters for your MCMC run (p0) 
init_params={
            "logmass_stell": 0.2672091787012066,  
            "incl": 73.29077893244971, 
            "Rc": 42.56025926929232,  
            "pp": 2.28775049147713, 
            "pa": 11.716316094540533,    
            "logmass": -5.202798478892286, 
            "Rin": 41.33932507247196,   
            "xoff": 0.002344042320019,   
            "yoff": -0.0439296398276957,  
            "vsys": 4.235875794042619, 
            "tatm": 299.8332659911651, 
            #"new_parameter_name1":new_parameter_initial_value,
            #..., 
}

# the widths of the parameter distributions (AKA "sigma" in previous mcmc scripts) 
params_uncertainty={
            "logmass_stell": 0.05,
            "incl": 1,
            "Rc": 0.5,
            "pp": 0.1,
            "pa": 1,
            "logmass": 0.05,
            "Rin": 1.0,
            "xoff": 0.05,
            "yoff": 0.05, 
            "vsys": 0.1,
            "tatm": 3,
            #"new_parameter_name1":new_parameter_uncertainty,
            #...,
}

param_priors={
            "logmass_stell":[-1, 1],
            "incl":[0, 90], # you should only ever restrict this range; do not make it larger as it's physically unreasonable
            "Rc":[0, 200],
            "pp":[-5, 5],
            "pa":[0, 360],
            "logmass":[-10, 0],
            "Rin":[0,100],
            "xoff":[-1,1],
            "yoff":[-1,1],
            "vsys":[3,15],
            "tatm":[0,500],
            #"new_parameter_name1":[new_parameter_lower_bound,new_parameter_upper_bound],
            #...,
}

name_map={
         'logmass_stell':r'log(M$_{star}$) [log(M$_{\rm \odot}$)]', 
         'incl':r'$incl$ [$^{\circ}$]', 
         'Rc':r'$R_{c}$ [au]', 
         'pp':r'$pp$', 
         'pa':r'PA [$^{\circ}$]', 
         'logmass':r'log(M$_{disk}$) [log(M$_{\rm \odot}$)]',
         'Rin':r'$R_{in} [au]$',
         'xoff':r'x$_{off}$', 
         'yoff':r'y$_{off}$',
         'vsys':r'$v_{sys} [km s^{-1}]$',
         'tatm':r'$T_{atm}$ [K]',
         #"new_parameter_name1":r'$newparametername_{1}',
         #...,
}

#################################################

# MCMC chain configurations
mcmc_hyper_params = {
            "nwalkers": 26, 
            "nsteps": 4000,
            "ndim": len(init_params),
            "restart": False
}
#################################################

# NEED this print statement to have dmr-script read in variables
print(f"{DATAFILE} {PIXEL_SIZE} {INPUT_MODEL_DMR} {INPUT_RESID_DMR} {DMR_OUTPUTS} {DMR_PREFIX} {mcmc_hyper_params['nwalkers']}") 

#################################################

# the code below will only be executed if this file is run directly (as opposed to calling it from another file as done in mcmc_trigger.py and dmr.py
# to run file directly enter python p0.py in the engine/. directory 
if __name__ == '__main__':
   # only add print statements here ... this is your playground 
   print(init_params)
   print(param_order)
