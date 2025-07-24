from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
# FILES WILL BE UNTARRED IN THE DIRECTORY THIS FILE IS IN 


def plt_iso():

    """
    uses iso files from download_UBVRI.py to create curves of constant mass and varying age
    """
   
    iso_file = input("Enter absolute path to the downloaded Isochrone files (must be .txz): ")
    while os.path.exists(iso_file) != True:
        iso_file = input("Not a valid file path. Enter absolute path to the downloaded Isochrone files (must be .txz): ")
            

    #untar the file and store needed files in current_directory 
    os.system("tar -xvf " + iso_file)

    #difference between a tar and untar file is the .tkz
    untar_iso_path = iso_file[:-4]

    #ask user for metallicity isochrone curve they want (note should be the same as the one selected for EEP)
    print()
    print("1. [Fe/H] = -4.00 \t 2. [Fe/H] = -3.50 \t 3. [Fe/H] = -3.00")
    print("4. [Fe/H] = -2.50 \t 5. [Fe/H] = -2.00 \t 6. [Fe/H] = -1.75")
    print("7. [Fe/H] = -1.50 \t 8. [Fe/H] = -1.25 \t 9. [Fe/H] = -1.00")
    print("10. [Fe/H] = -0.75 \t 11. [Fe/H] = -0.50 \t 12. [Fe/H] = -0.25")
    print("13. [Fe/H] = +0.00 \t 14. [Fe/H] = +0.25 \t 15. [Fe/H] = +0.50")

    user_metallicity = int(input("What [Fe/H] would you like: "))

    valid = {1:'4.00',2:'3.50',3:'3.00',4:'2.50',5:'2.00',6:'1.75',7:'1.50',8:'1.25',9:'1.00',10:'0.75',11:'0.50',12:'0.25',13:'0.00',14:'0.25',15:'0.50'}

    i = False
    while i != True: 
        if user_metallicity not in valid:
            i = False
            print("invalid selction, try again ")
            user_metallicity = int(input("What [Fe/H] would you like: "))
        else:
            i = True

    if user_metallicity < 13:
        index = 'm'+valid[user_metallicity]
    elif user_metallicity >= 13:
        index = 'p'+valid[user_metallicity]
    
    print(untar_iso_path[-30:]+"/MIST_v1.2_feh_"+index+"_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd")
    datai = ascii.read(untar_iso_path[-29:]+"/MIST_v1.2_feh_"+index+"_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd")


    logli = np.array(datai['col7'])
    logti = np.array(datai['col5'])
    massi = np.array(datai['col3'])
    agei = np.array(datai['col2'])

    #age ranges in log years
    max_age = max(agei)  
    min_age = agei[0]          

    #mass ranges in solar masses
    max_mass = max(massi)
    min_mass = massi[0]

    #begin loop that allows user to plot many constant mass curves using the data from one file
    user_dec = 'yes'
    while user_dec != 'no':

        #ask user to enter age ranges for constant mass curve
        user_age_min = float(input("Enter minimum age, in log years, for constant mass curve (range: "+str(min_age)+" (100,000 years) ~ "+str(max_age)+ " (10 Gyr)): "))
        i = False
        while i != True:
            if user_age_min > max_age or user_age_min < min_age:
                print("Invalid entry, try again.")
                user_age_min = float(input("Enter minimum age, in log years, for constant mass curve (range: "+str(min_age)+" (100,000 years) ~ "+str(max_age)+ " (10 Gyr)): "))
                i = False
            else:
                i = True
        
        user_age_max = float(input("Enter maximum age, in log years, for constant mass curve (range: "+str(user_age_min)+" (100,000 years) ~ "+str(max_age)+ " (10 Gyr)): "))
        j = False
        while j != True:
            if user_age_max > max_age or user_age_max < user_age_min:
                print("Invalid entry, try again.")
                user_age_max = float(input("Enter maximum age, in log years, for constant mass curve (range: "+str(user_age_min)+" (100,000 years) ~ "+str(max_age)+ " (10 Gyr)): "))
                j = False
            else:
                j = True

        
        #ask user to enter mass ranges for constant mass curves
        user_mass_min = float(input("Enter minimum mass, in solar massess, for constant mass curve (range: "+str(min_mass)+" ~ "+str(max_mass)+") : "))
        i = False
        while i != True:
            if user_mass_min > max_mass or user_mass_min < min_mass:
                print("Invalid entry, try again.")
                user_mass_min = float(input("Enter minimum mass, in solar massess, for constant mass curve (range: "+str(min_mass)+" ~ "+str(max_mass)+") : "))
                i = False
            else: 
                i = True
        
        user_mass_max = float(input("Enter maximum mass, in solar massess, for constant mass curve (range: "+str(min_mass)+" ~ "+str(max_mass)+") : "))
        j = False
        while j != True:
            if user_mass_max > max_mass or user_mass_max < min_mass:
                print("Invalid entry, try again.")
                user_mass_min = float(input("Enter maximum mass, in solar massess, for constant mass curve (range: "+str(min_mass)+" ~ "+str(max_mass)+") : "))
                j = False
            else:
                j = True

        #gather data that fit user criterion
        loci1 = np.where((agei>user_age_min)&(agei<user_age_max)&(massi>user_mass_min)&(massi<user_mass_max))
        i1 = plt.plot(logti[loci1], logli[loci1], color = 'red', linestyle = 'dashed', label = input("Input label for isochrone curve: "))


        user_dec = input("Would you like to plot another constant mass curve? Please enter 'yes' or 'no': ")

    
    return 

