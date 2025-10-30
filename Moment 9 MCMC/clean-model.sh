#!/bin/bash

echo "unique_id: $1"
UID=$1
echo "UID: ${UID}"
fits in=model_DMR_LB$1.uvfits out=model_DMR_LB$1.vis op=uvin options=varwt
fits in=model_DMR_SB$1.uvfits out=model_DMR_SB$1.vis op=uvin options=varwt

invert vis=model_DMR_SB$1.vis,model_DMR_LB$1.vis map=modelmom1$1.mp beam=modelmom1$1.bm imsize=512 cell=0.03 robust=2

clean map=modelmom1$1.mp beam=modelmom1$1.bm out=modelmom1$1.cl cutoff=0.01 niters=10000

restor beam=modelmom1$1.bm map=modelmom1$1.mp model=modelmom1$1.cl out=modelmom1$1.cm

# Change names of output file
fits in=modelmom1$1.cm out=modelmom1_$1.fits op=xyout

rm -rf model_DMR_?B$1.* modelmom1$1.*
