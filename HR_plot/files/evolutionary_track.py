import os
import math as m 
import numpy as np
import matplotlib.pyplot as plt
from fastnumbers import isfloat
from astropy.io import ascii
    
def plot_eep(): 
        
    eep_file = input("Enter absolute path to the downloaded Evolutionary Track files (must be .txz): ")
    while os.path.exists(eep_file) != True:
        eep_file = input("Not a valid file path. Enter absolute path to the downloaded Evolutionary Track files (must be .txz): ")
        
            

    #untar the file and store needed files in current_directory 
    os.system("tar -xvf " + eep_file)

    #difference between a tar and untar file is the .tkz
    untar_iso_path = eep_file[:-4]
    path = untar_iso_path[-43:]
    # Insert the directory path in here
    
    # Extracting all the contents in the directory corresponding to path
    l_files = os.listdir(path)
    
    arr_clean = []  
    j = 0
    #gets rid of of 3 doc files
    while j < (len(l_files)):
        if l_files[j] == 'README_tables.pdf' or l_files[j] == 'README_overview.pdf' or l_files[j] == "._README_overview.pdf" or l_files[j] == "._README_tables.pdf":
            j = j + 1
        else:
            arr_clean.append(l_files[j])
            j = j + 1

    print(arr_clean) 
    # sort the mass files in increasing order and show user
    # Each file is named according to the star mass its data corresponds to. The following makes an array composed of only 
    # the mass digits associated with each file -- this is needed to determine which file the user wants to interact with  
    v_selec = sorted(arr_clean)
    file_ints_str = []
    for file in v_selec: 
        print(file + ' Solar Masses: ' + str(float(file[0:5])/100))
        file_ints_str.append(file[0:5])

    print()
    print("INSTRUCTIONS: Input must be of the form ######. For example if you wish to input a 255.05 solar mass star then enter 25505. Likewise if you choose to enter a 1.10 solar mass star enter 00110.")
    print()

    # ask for LOW interp bound; this gives us the minimum mass we want to graph (and posssibly interpolate for)
    min_interp_dec = input("Please input the MINIMUM eep mass curve you want to highlight: ")
    while isfloat(min_interp_dec) != True or min_interp_dec < '00010' or min_interp_dec > '30000' or len(min_interp_dec) >= 6 or len(min_interp_dec) < 5:
        min_interp_dec = input("Not a valid input. Please input the MINIMUM eep mass curve you want to highlight (range is 00010 - 30000 solar masses): ")
       
    # if min_interp_dec is in the array of files then no need to interpolate just use existing data
    if min_interp_dec in file_ints_str:
        
        dataminlow = ascii.read(path+'/'+min_interp_dec+"M.track.eep")

        logt1 = np.array(dataminlow['col12'])
        logl1 = np.array(dataminlow['col7'])
        age1 = np.array(dataminlow['col1'])
        min_age_minlow = age1[0]
        max_age_minlow = age1[len(age1)-1]

        user_age_min_i = float(input("Enter minimum age for lower bound of highlighted region (" + str(min_age_minlow) + " years - " + str(max_age_minlow) + " years): "))
        while user_age_min_i < float(min_age_minlow) or isfloat(user_age_min_i) == False:
            user_age_min_i = float(input("Not a valid age selection. Enter minimum age for lower bound of highlighted region (" + str(min_age_minlow) + " years - " + str(max_age_minlow) + " years): "))

        user_age_max_i = float(input("Enter maximum age for lower bound of highlighted region (" + str(user_age_min_i) + " years - " + str(max_age_minlow) + " years): "))
        while user_age_max_i > max_age_minlow or user_age_max_i < user_age_min_i or isfloat(user_age_max_i) == False:
            user_age_max_i = float(input("Not a valid age selection. Enter maximum age for lower bound of highlighted region (" + str(user_age_min_i) + " years - " + str(max_age_minlow) + " years): "))
    
        locminlow = np.where((age1>user_age_min_i)&(age1<user_age_max_i))
        low_temp_arr = logt1[locminlow]
        low_lum_arr = logl1[locminlow]
         

    # min_interp_dec not an existing file so interpolate
    elif min_interp_dec not in file_ints_str:
        print()
        print('Mass selected has no existing data file. Begin interpolation routine for low-bound curve.')
        print()

        # find what existing files the min interp bound is between
        i = 0
        while i < len(file_ints_str) and file_ints_str[i] < min_interp_dec:
            i = i + 1
            
        min_max = file_ints_str[i]
        min_min = file_ints_str[i-1]
        
        print("Min-Low file selected is: " + min_min + "M.track.eep, your input was " + min_interp_dec +".")

        dataminlow = ascii.read(path+'/'+min_min+"M.track.eep")

        logt1 = np.array(dataminlow['col12'])
        logl1 = np.array(dataminlow['col7'])
        age1 = np.array(dataminlow['col1'])
        min_age_minlow = age1[0]
        max_age_minlow = age1[len(age1)-1]

        # ask user what age they want interp curves to cover (note: the age range given for LOW INTERP will be the same as the one assigned for the HIGH INTERP range; we 
        # will then reduce the longer of the two arrays by array slicing until they are the same size)
        user_age_min_i = float(input("Enter minimum age for lower bound of highlighted region (" + str(min_age_minlow) + " years - " + str(max_age_minlow) + " years): "))
        while user_age_min_i < min_age_minlow or isfloat(user_age_min_i) == False:
            user_age_min_i = float(input("Not a valid age selection. Enter minimum age for lower bound of highlighted region (" + str(min_age_minlow) + " years - " + str(max_age_minlow) + " years): "))

        user_age_max_i = float(input("Enter maximum age for lower bound of highlighted region (" + str(user_age_min_i) + " years - " + str(max_age_minlow) + " years): "))
        while user_age_max_i > max_age_minlow or user_age_max_i < user_age_min_i or isfloat(user_age_max_i) == False:
            user_age_max_i = float(input("Not a valid age selection. Enter maximum age for lower bound of highlighted region (" + str(user_age_min_i) + " years - " + str(max_age_minlow) + " years): "))
        
        # gather data for minlow arrays; this restricts the range of values we are considering to the age range given by the user
        locminlow = np.where((age1>user_age_min_i)&(age1<user_age_max_i))

        # temp and lum array for minlow with range restrictions as given by user
        temp_minlow = logt1[locminlow]
        lum_minlow = logl1[locminlow]

        #  This completes accessing the minimum interp data points for low bound. We now need the minimum high values to then interpolate between 
        #  both sets of data.
        #  The below accomplishes the second portion of the interpolation routine for the LOW boundary.
               
        print("Max-low file selected is: " + min_max + "M.track.eep, your input was " + min_interp_dec +".")

        datamaxlow = ascii.read(path+'/'+min_max + "M.track.eep")

        logt1 = np.array(datamaxlow['col12'])
        logl1 = np.array(datamaxlow['col7'])
        age1 = np.array(datamaxlow['col1'])

        # gather data for maxlow arrays; recall we use the same age as that given for the min_low array
        locmaxlow = np.where((age1>user_age_min_i)&(age1<user_age_max_i))

        # temp and lum for maxlow
        temp_maxlow = logt1[locmaxlow]
        lum_maxlow = logl1[locmaxlow]

        # slice arrays so that they have the same size; we need to do this so that there is a one to one interpolation 
        # betwen points in the array          
        while len(temp_minlow) != len(temp_maxlow):
            if len(temp_maxlow) > len(temp_minlow):
                temp_maxlow = temp_maxlow[1:]
            else: 
                temp_minlow = temp_minlow[1:]

        while len(lum_minlow) != len(lum_maxlow):
            if len(lum_maxlow) > len(lum_minlow):
                lum_maxlow = lum_maxlow[1:]
            else:
                lum_minlow = lum_minlow[1:]
 
        tot_steps = int(m.ceil((float(min_max)/100 - float(min_min)/100)/0.01))
        float_user_dec_min = float(min_interp_dec)/100
        start = float(min_min)/100

        # arrays are now the same size so they are ready for interpolation. We add 0.01 to min_min until we reach min_interp_dec. 
        # Recall min_min and min_max are the largest mass file less than min_interp_dec and smallest mass file larger than min_interp_dec
        # respectively. Thus the below determines how many steps of 0.01 we need to take (from min_min) to get as close as possible
        # to the min_interp_dec. In doing so we know which index to consider when we perform interpolation for each ith index (see
        # next while loop with k < len(temp_maxlow) condition).        

        i = 0
        while start <= float_user_dec_min:
            start = start + 0.01
            i = i + 1 
        
        # 'i' keeps track of how many steps we have to take from the base case (min_min) to get to the user input min_interp_dec.
        # The two while loops below complete the interpolation routine for the user input lower bound of the highlighted region, they 
        # do so by using linspace as a mock interpolation routine. The linspace is used to evenly divide the difference
        # of the ith index of the temp_minlow and temp_maxlow arrays. 
        # i = i - 1
        k = 0
        low_temp_arr = []
        while k < len(temp_maxlow):
            x = np.linspace(temp_minlow[k], temp_maxlow[k], tot_steps+2)
            low_temp_arr.append(x[i])
            k  = k + 1

        y = 0
        low_lum_arr = []
        while y < len(temp_maxlow):
            x = np.linspace(lum_minlow[y], lum_maxlow[y], tot_steps+2)
            low_lum_arr.append(x[i])
            y = y + 1

    # ask for HIGH interp bound; this gives us the maximum mass we want to interpolate to 
    max_interp_dec = input("Please input the MAXIMUM eep mass curve you want to highlight: ")
    while isfloat(max_interp_dec) != True or max_interp_dec < min_interp_dec or len(max_interp_dec) >= 6 or len(max_interp_dec) < 5:
        max_interp_dec = input("Not a valid input. Please input the MAXIMUM eep mass curve you want to highlight (range is "+str(min_interp_dec)+" - 30000 solar masses): ")
     

    # if High interp is in the array of files then no need to interpolate just use existing data
    if max_interp_dec in file_ints_str:
        
        print("Min-high file selected is: " + max_interp_dec + "M.track.eep, your input was " + max_interp_dec +".")

        dataminhigh = ascii.read(path+'/' + max_interp_dec+"M.track.eep")

        logt1 = np.array(dataminhigh['col12'])
        logl1 = np.array(dataminhigh['col7'])
        age1 = np.array(dataminhigh['col1'])
    
        # gather data for minhigh arrays. Note the age used is the same as given for the low interp.
        locminhigh = np.where((age1>user_age_min_i)&(age1<user_age_max_i))

        # temp and lum array for high interp
        high_temp_arr = logt1[locminhigh]
        high_lum_arr = logl1[locminhigh]


    elif max_interp_dec not in file_ints_str:
        print()
        print('Mass selected has no existing data file. Begin interpolation routine for high-bound curve.')
        print()

        # FOR HIGH INTERP: find the solar mass files the min interp is inbetween
        i = 0
        while i < len(file_ints_str) and file_ints_str[i] < max_interp_dec:
            i = i + 1
            
        max_max = file_ints_str[i]
        max_min = file_ints_str[i-1]
    
        # isolate min file for HIGH INTERP
        print("Min-high file selected is: " + max_min + "M.track.eep, your input was " + max_interp_dec +".")

        dataminhigh = ascii.read(path+'/' + max_min+"M.track.eep")

        logt1 = np.array(dataminhigh['col12'])
        logl1 = np.array(dataminhigh['col7'])
        age1 = np.array(dataminhigh['col1'])
    
        # gather data for minhigh arrays
        locminhigh = np.where((age1>user_age_min_i)&(age1<user_age_max_i))

        # temp and lum array for minhigh
        temp_minhigh = logt1[locminhigh]
        lum_minhigh = logl1[locminhigh]

        # This completes accessing the minimum high data points. We now need the maximum high values to then interpolate 
        # both sets of data. The below accomplishes the second portion of the interpolation routine for the HIGH boundary.

        print("Max-high file selected is: " + max_max + "M.track.eep, yourr input was " + max_interp_dec +".")

        datamaxhigh = ascii.read(path+'/' + max_max+"M.track.eep")

        logt1 = np.array(datamaxhigh['col12'])
        logl1 = np.array(datamaxhigh['col7'])
        age1 = np.array(datamaxhigh['col1'])

        # gather data for maxhigh arrays
        locmaxhigh = np.where((age1>user_age_min_i)&(age1<user_age_max_i))

        # temp and lum for maxlow
        temp_maxhigh = logt1[locmaxhigh]
        lum_maxhigh = logl1[locmaxhigh]

        # ensure minhigh and maxhigh have arrays of the same size for all dimensions
        while len(temp_minhigh) != len(temp_maxhigh):
            if len(temp_maxhigh) > len(temp_minhigh):
                temp_maxhigh = temp_maxhigh[1:]
            else: 
                temp_minhigh = temp_minhigh[1:]

        while len(lum_minhigh) != len(lum_maxhigh):
            if len(lum_maxhigh) > len(lum_minhigh):
                lum_maxhigh = lum_maxhigh[1:]
            else:
                lum_minhigh = lum_minhigh[1:]

        tot_steps = int((float(max_max)/100 - float(max_min)/100)/0.01)
        float_user_dec_max = float(max_interp_dec)/100
        start = float(max_min)/100

        i = 0 
        while start < float_user_dec_max:
            start = start + 0.01
            i = i + 1

        #i = i - 1
        k = 0
        high_temp_arr = []
        while k < len(temp_maxhigh):
            x = np.linspace(temp_minhigh[k], temp_maxhigh[k], tot_steps+2)
            high_temp_arr.append(x[i])
            k  = k + 1

        y = 0
        high_lum_arr = []
        while y < len(temp_maxhigh):
            x = np.linspace(lum_minhigh[y], lum_maxhigh[y], tot_steps+2)
            high_lum_arr.append(x[i])
            y = y + 1


    x1 = low_temp_arr
    y1 = low_lum_arr
    x2 = high_temp_arr
    y2 = high_lum_arr

    print([len(x1),len(x2),len(y1),len(y2)])
    plt.plot(x1, y1, 'x', label = input("Input label for lower bound: "))
    plt.plot(x2, y2, 'x', label = input("Input label for upper bound: "))

    fill_dec = input("Would you light to highlight the region between these curves? (yes or no) ")
    while fill_dec == 'yes':
        plt.fill(np.append(x1, x2[::-1]), np.append(y1, y2[::-1]))
        return
        
    return 
    

