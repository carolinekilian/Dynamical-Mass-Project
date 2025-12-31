INSTRUCTIONS: 

Place all unzipped/untarred stellar evolutionary track data in the same directory as master.py and evolutionary_track.py 

The hrcode can ingest data from the MESA MIST, BHAC15, PARSEC1.2S, Feiden - Magnetic, and Feiden Non-Magnetic
stellar models. 

Links to each dataset: 
    
    1. MESA MIST DATA: https://waps.cfa.harvard.edu/MIST/model_grids.html
    - Download any file under "EEP Tracks" (including v/vcrit=0.4 and v/vcrit=0.0)
    
    - Directory Naming convention: metallicity is given by the stuff after the p and m that follows feh
    Ex: hrplot/MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_EEPS has metallicity -0.25
    Ex: hrplot/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS has metallicity +0.00

    - File Naming convention: MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/00010M.track.eep
      Denotes track for a star of solar mass = 0.01  

    2. BHAC15 (Baraffe et. al, 2015): https://zenodo.org/records/15729114 
    - Use the "Download all" button (2.8 MB)

    3. PARSEC1.2: https://stev.oapd.inaf.it/PARSEC/tracks_v12s.html 
    - Use the "Get all grids" button (255.6 MB)
    
    - Directory Naming convention: Metallicity is given by the numbers following the Z in the file name. 
        Ex: file hrplot/all_tracks_Pv1.2s/Z0.0002Y0.249.zip has tracks with 
        metallicity, Z=0.0002, and helium mass fraction, Y=0.249. 
        Recall the hydrogen mass fraction, X = 1 - Z - Y
    
    - File Name convention: Z0.001Y0.25/Z0.001Y0.25OUTA1.77_F7_M000.700.DAT
        Denotes tracks of a star of Z = 0.0002, Y=0.25, and Solar mass of 0.7
    - lots of tracks for low metallicities

    4. Feiden - Magnetic: https://github.com/gfeiden/MagneticUpperSco/tree/master/models/trk/mag
    - Download the file titled: all__GS98_p000_p0_y28_mlt1.884_Beq.tgz (7.4 MB)
    
    - Directory Naming convention: the numbers after the y are the helium mass fraction they are computed using the 
      Z and X at the top of each file. B/c y is the same for all files, all files have the same metallicity
      of Z = 0.018. 
    
    - File Naming Convetion hrplot/all_GS98_p000_p0_y28_mlt1.884 - Feiden Non-Magnetic/m0090_GS98_p000_p0_y28_mlt1.884.trk
        Denotes star of solar mass 0.09 and Y = 0.28
    

    5. Feiden - Non-Magnetic: https://github.com/gfeiden/MagneticUpperSco/tree/master/models/trk/std
    - Download the file titled all_GS98_p000_p0_y28_mlt1.884.tgz (41.6 MB)
    - Same directory and file naming 