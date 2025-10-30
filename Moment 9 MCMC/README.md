The most recent version of this code is in final_mcmc_code.py (it is running HD121617). You will also need clean-model.sh and datamomscript.py.

datamomscript.py
-run before running the mcmc chain to create a moment 9 file and an error file for your data. You should only need to change the file names to be accurate for your disk.

clean-model.sh
-run every mcmc loop to clean the created model file in the same process used for the data. *MAKE SURE TO CHANGE THE NUMBERS* so it accurately reflects the cleaning done with the data.

The code is currently set up to create two different wavelength files from the model ad then combine them in clean-model.sh. This it to mitigate the "too many antennas" error. If you do not need to do this, change final_mcmc_code to call the visibilites only once and remove the mosaicing from clean-model.sh.
