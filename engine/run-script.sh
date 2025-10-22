#!/bin/bash
python_output=$(python3 p0.py)
read -r DATAFILE PIXEL_SIZE INPUT_MODEL_DMR INPUT_RESID_DMR OUTPUT_DIR DMR_PREFIX NWALKERS <<< "$python_output"

mcmc_file="$1"
mpirun -np "$NWALKERS" nice python3 "$mcmc_file"
