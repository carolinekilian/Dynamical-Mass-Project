DO NOT MOVE FILES OR RENAME DIRECTORIES - the code assumes the current directory structure 
IF YOU WANT TO RUN ANY GIVEN FILE YOU MUST DO SO FROM THE CURRENT WORKING DIRECTORY (i.e engine/.) - do not run MCMC plotting files from MCMC_OUTPUTS
##########################
### SUGGESTED WORKFLOW ###
##########################

PHASE 1: Place data in directory and ensure that your disk parameter initialization is a good starting point for your MCMC run. 
1. Place all of you weight-corrects .uvfits files along with your fits files in the CAS_OUTPUTS directory. 
2. Update p0.py to reflect system/disk parameters and file names in CASA_OUTPUTS
3. Run dmr.py using the init_params to make a disk object, raytrace it, and generate dmr. 
   - Inspect the spectrum plot, updating the initial parameters as needed and rerunning dmr.py to ensure a better match between model and data. 
     You will not need to manually delete any files as they will be overwritten. Simply update p0.py and rerun dmr.py.
   - A "good initialization" consists of aligned spectrum peaks. Data and model flux peaks should be within a multiple of three. 
    
4. Run data_in_miriad.sh, model_in_miriad.sh, and resid_in_miriad.sh to make sure channel maps look similiar and flux ranges are in same ball park. 
5. Run MCMC using: ./run-script.sh mcmc_trigger.py

PHASE 2: Monitor MCMC
6. Generate walker plots using: python MCMC_OUTPUTS/walker_plot.py 
   - You should observe walkers converging on a region/sliver of parameter space per disk parameter. 
   - TBH I think this is a pretty diagnostic tool ... just make sure that most walkers are climbing to larger lnprobs 
7. Generate corner plot using: python MCMC_OUTPUTS/corner.py
   - Make sure your parameters aren't creeping up to their prior bounds -- unless it's the disk mass, when the gas is optically thick the gas flux doesn't increase. If they are 
     restart the chain with widended priors. 
8. Generate spectrum plot with best-fit parameters: python dmr.py
   - Just make sure you model is looking at least as good as the model you generated in step 3 of PHASE 1. 
9. Run data_in_miriad.sh, model_in_miriad.sh, and resid_in_miriad.sh. Eventually your data and model channel maps will look similiar and residuals will be noise-like. 

#######################
### FILE DESCRIPTION ## 
#######################
----------------------------------------------------------------------------
 
- p0.py: contains all parameter, files, and direcotry naming conventions; you should only need to modify this file for any given disk with 
         any given set of parameters

         ex usage: executes all commands in playground (section at bottom of file) 
         python p0.py 

----------------------------------------------------------------------------

disk.py, raytrac.py, and co.dat are files inherited from Kevin's codebase. They should need no further edits. Hydrostatic equilibirum, sigbound bounds, and freeze out temperature updates already performed. Refer to this link for an in depth discussion of Kevin's code: https://github.com/kevin-flaherty/disk_model3/blob/master/Guide.ipynb

NOTE: You should never need to manually run these files (i.e python <filename>) 
- disk.py: makes disk object, basically this is a big function that describes the temperature and density of gas within in a debris disk as a function of radius 
- raytrace.py: computes how light interacts with disk object
- co.dat: data required to assign 'assets' (i.e gas) in the disk medium 

-----------------------------------------------------------------------------

Data-Model-Residual (DMR) Files: used to generate diagnostics of disk model given a set of disk parameters

- dmr.py: creates disk object given MCMC best-fit parameters or initializations parameters using disk.py and co.dat, raytraces disk object 
          using raytrace.py, and triggers dmr-script and specfig.py to make spectrum plot
          
          ex usage: using the init_params to make a disk object, raytrace it, and generate dmr 
          run dmr (yes or no): yes 
          p0 or MCMC: p0 
          
          ex usage: using the MCMC best-fit parameters to make a disk object, raytrace it, and generate dmr
          run dmr (yes or no): yes 
          p0 or MCMCM: MCMC

          ex usage: printing the mcmc best-fit parameters
          run dmr (yes or no): no  
  
NOTE: You should never need to manually run dmr-script.sh and specfig.py as they're triggered in dmr.py
- dmr-script.sh: uses galario to convert the image-domain raytraced-disk-object into synthetic vissibilities and compares to casa-generated uvfits 
                 and MIRIAD to make spectrum plot data.   
- specfig.py: cleans miriad-generated spectrum plot data and produces matplotlib plot

ex usage for the three .sh files below: 
./<filename>

- residuals_in_miriad.sh: displays residual channel maps in miriad
- data_in_miriad.sh: displays data channel maps in miriad
- model_in_miriad.sh: displays model channel maps in miriad

------------------------------------------------------------------------------

Markov Chain Monte Carlo (MCMC) Files:
- mcmc_trigger.py: use this to trigger your mcmc run; samples parameters and uses them to generate a disk object which is ultimately compared to the data. You should never need to run this 
                   file manually as it's triggered in run-script.sh.
- MCMC_OUTPUTS: contains files that create MCMC diagnostic plots (walker and corner plots); this is also where your parameter samples (i.e the .csv file)
                and parallax samples (i.e the .txt) file are stored
                ex usage: python MCMC_OUTPUTS/corner_plot.py
                ex usage: python MCMC_OUTPUTS/walker_plot.py
- run-script.sh: used to trigger the mcmc runs so that it's distributed across several CPUs
                ex usage: trigger mcmc
                NOTE: I always trigger mcmc with as many CPUs as walkers.  
                ./run-script mcmc_trigger.py

------------------------------------------------------------------------------

CASA_DATA: place all of your casa-generated, weight corrected (i.e var_vis processed) .uvfits and fits files here 

------------------------------------------------------------------------------

DMR_OUTPUTS: contains all dmr.py, dmr-script.sh, and specfig.py outputs

------------------------------------------------------------------------------
