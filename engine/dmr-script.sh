#!/bin/bash

python_output=$(python3 p0.py)
read -r DATAFILE PIXEL_SIZE INPUT_MODEL_DMR INPUT_RESID_DMR OUTPUT_DIR DMR_PREFIX NWALKERS <<< "$python_output"

echo "datafile: $DATAFILE"
echo "pixel size: $PIXEL_SIZE"
echo "input resid DMR file: $INPUT_RESID_DMR"
echo "input model DMR file: $INPUT_MODEL_DMR"
echo "dmr output directory: $OUTPUT_DIR"
echo "dmr prefix: $DMR_PREFIX"
DMR_GALARIO_PREFIX=${DMR_PREFIX}_GALARIO
DMR_MIRIAD_PREFIX=${DMR_PREFIX}_MIRIAD
rm -rf $OUTPUT_DIR/*$DMR_GALARIO_PREFIX*
rm -rf $OUTPUT_DIR/*$DMR_MIRIAD_PREFIX*

echo "BEGINNING GALARIO PROCESSING"
fits in=$DATAFILE out=${DMR_GALARIO_PREFIX}_datavis.vis op=uvin options=varwt
fits in=$INPUT_RESID_DMR out=${DMR_GALARIO_PREFIX}_residvis.vis op=uvin options=varwt
fits in=$INPUT_MODEL_DMR out=${DMR_GALARIO_PREFIX}_modvis.vis op=uvin options=varwt

invert vis=${DMR_GALARIO_PREFIX}_datavis.vis map=${DMR_GALARIO_PREFIX}_data.mp beam=${DMR_GALARIO_PREFIX}_data.bm imsize=256 cell=$PIXEL_SIZE robust=0.5
invert vis=${DMR_GALARIO_PREFIX}_residvis.vis map=${DMR_GALARIO_PREFIX}_resid.mp beam=${DMR_GALARIO_PREFIX}_resid.bm imsize=256 cell=$PIXEL_SIZE robust=0.5 options=systemp #mosaic,mfs
invert vis=${DMR_GALARIO_PREFIX}_modvis.vis map=${DMR_GALARIO_PREFIX}_model.mp beam=${DMR_GALARIO_PREFIX}_model.bm imsize=256 cell=$PIXEL_SIZE robust=0.5 options=systemp #mosaic,mfs

clean map=${DMR_GALARIO_PREFIX}_data.mp beam=${DMR_GALARIO_PREFIX}_data.bm out=${DMR_GALARIO_PREFIX}_data.cl cutoff=0 niters=1
clean map=${DMR_GALARIO_PREFIX}_resid.mp beam=${DMR_GALARIO_PREFIX}_resid.bm out=${DMR_GALARIO_PREFIX}_resid.cl cutoff=0 niters=1      
clean map=${DMR_GALARIO_PREFIX}_model.mp beam=${DMR_GALARIO_PREFIX}_model.bm out=${DMR_GALARIO_PREFIX}_model.cl cutoff=0 niters=1

restor beam=${DMR_GALARIO_PREFIX}_data.bm map=${DMR_GALARIO_PREFIX}_data.mp model=${DMR_GALARIO_PREFIX}_data.cl out=${DMR_GALARIO_PREFIX}_data.cm
restor beam=${DMR_GALARIO_PREFIX}_resid.bm map=${DMR_GALARIO_PREFIX}_resid.mp model=${DMR_GALARIO_PREFIX}_resid.cl out=${DMR_GALARIO_PREFIX}_resid.cm
restor beam=${DMR_GALARIO_PREFIX}_model.bm map=${DMR_GALARIO_PREFIX}_model.mp model=${DMR_GALARIO_PREFIX}_model.cl out=${DMR_GALARIO_PREFIX}_model.cm

fits in=${DMR_GALARIO_PREFIX}_resid.cm out=${DMR_GALARIO_PREFIX}_dmr_resid.fits op=xyout
fits in=${DMR_GALARIO_PREFIX}_model.cm out=${DMR_GALARIO_PREFIX}_dmr_model.fits op=xyout
echo "FINISHED GALARIO PROCESSING"

echo "GENERATING SPECTRUM FILES"
imspec in=${DMR_GALARIO_PREFIX}_data.cm device=/xs region=arcsec,box'(-3,-3,3,3)' options=list log=${DMR_MIRIAD_PREFIX}_data_spec.txt
imspec in=${DMR_GALARIO_PREFIX}_resid.cm device=/xs region=arcsec,box'(-3,-3,3,3)' options=list log=${DMR_MIRIAD_PREFIX}_resid_spec.txt
imspec in=${DMR_GALARIO_PREFIX}_model.cm device=/xs region=arcsec,box'(-3,-3,3,3)' options=list log=${DMR_MIRIAD_PREFIX}_model_spec.txt
echo "FINISHED GENERATING SPECTRUM FILES"
mv *${DMR_PREFIX}* $OUTPUT_DIR
