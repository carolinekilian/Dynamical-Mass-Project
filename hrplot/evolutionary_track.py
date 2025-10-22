import os
import math
import numpy as np
import matplotlib.pyplot as plt
from fastnumbers import isfloat, isint
from astropy.io import ascii
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True

def get_valid_input(prompt, error_prompt, command, var_type):
    """
    Helper function to get valid input for x and y points.
    """
    if command:
        value=command[var_type]
        if not isint(value) and not isfloat(value):
            raise ValueError(f"Invalid {var_type} in: {command}")
    else:    
        value = input(prompt)
        while not isint(value) and not isfloat(value):
            value = input(error_prompt)
    return float(value)

convert_to_log = lambda x: np.log10(x) if x > 0 else 0 
# source: https://www.astro.princeton.edu/~gk/A403/constants.pdf
convert_to_bolometric_luminosity = lambda logL_sun: 4.8 - 2.5*logL_sun

def plot_point(fig, ax, color,command={}):
    """
    Function to plot a point with error bars on the HR diagram.
    """

    x_point = get_valid_input(
        "Please provide the log effective temperature [Kelvin]: Log(T_eff) = ",
        "Invalid input. Please provide a valid number for Log(T_eff): ",
        command,
        'temperature_kelvin'
    )

    x_point_err = get_valid_input(
        "Please provide the log error for effective temperature [Kelvin]: ",
        "Invalid input. Please provide a valid number for the error: ",
        command,
        'temperature_kelvin_err'
    )
    
    y_point = get_valid_input(
        "Please provide the log bolometric luminosity of the star: Log(L) = ",
        "Invalid input. Please provide a valid number for Log(L): ",
        command,
        'luminosity_solar_lum'
    )
    y_point_err = get_valid_input(
        "Please provide the log error for bolometric luminosity: ",
        "Invalid input. Please provide a valid number for the error: ",
        command,
        'luminosity_solar_lum_err'
    )
    if command:
        name=command['name']
    else:
        name = input("Please provide a name for this point: ")

    # Convert to log scale 
    x_point = convert_to_log(x_point)  
    x_point_err = np.abs(convert_to_log(x_point + x_point_err) - x_point) if x_point_err !=0 else 0
    print(f"x_point (log scale): {x_point} ± {x_point_err}")
    y_point = convert_to_log(y_point)
    y_point_err = convert_to_log(y_point + y_point_err) - y_point if y_point_err !=0 else 0
    print(f"y_point (log scale): {y_point} ± {y_point_err}")
    # convert to bolometric luminosity
    y_point = convert_to_bolometric_luminosity(y_point)
    y_point_err = np.abs(convert_to_bolometric_luminosity(y_point + y_point_err) - y_point) if y_point_err !=0 else 0
    print(f"y_point (bolometric luminosity): {y_point} ± {y_point_err}")

    ax.scatter([x_point], [y_point], color=color)
    ax.errorbar([x_point], [y_point], xerr=[x_point_err], yerr=[y_point_err], label=name, color=color, fmt='o')
    return fig, ax

def get_unique_color(available_colors, used_colors):
    if not available_colors:
        raise ValueError("No more unique colors available.")
    
    color = available_colors.pop(0)
    used_colors.add(color)

    return color, available_colors

def plot_line_of_constant_temperature(fig, ax, y_limits, command_temp):
    min_temp=command_temp['min_temp_kelvin']
    max_temp=command_temp['max_temp_kelvin']
    label=command_temp['label']

    #convert to log scale
    min_temp=convert_to_log(min_temp)
    max_temp=convert_to_log(max_temp)

    # get min and max of y axis
    y_min, y_max = y_limits
    # fill in between min and max temperature
    ax.fill_betweenx([y_min, y_max], min_temp, max_temp, color='lightgray', alpha=0.5, label=label)
    return fig, ax

def plot_line_of_constant_luminosity(fig, ax, x_limits, command_lum):
    min_lum=command_lum['min_lum_solar_lum']
    max_lum=command_lum['max_lum_solar_lum']
    label=command_lum['label']

    #convert to log scale
    min_lum=convert_to_bolometric_luminosity(convert_to_log(min_lum))
    max_lum=convert_to_bolometric_luminosity(convert_to_log(max_lum))

    # get min and max of x axis
    x_min, x_max = x_limits
    # fill in between min and max luminosity
    ax.fill_between([x_min, x_max], min_lum, max_lum, color='lightgray', alpha=0.5, label=label)
    return fig, ax

def plot_format(fig, ax, title, fontsize=30):
    plt.title(title, fontsize=fontsize)
    plt.xlabel(r'log ($T_{eff}$) [K]', fontsize=fontsize)
    plt.ylabel(r'log ($L_{bol}/L_{\odot}$)', fontsize=fontsize)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.grid()
    plt.show()

def file_cleanup(files):
    """Cleans up list of files by removing README and system files."""
    exclude = {'README_tables.pdf', 'README_overview.pdf', '.DS_Store'}
    return [f for f in files if f not in exclude]

def read_track_data(path, mass):
    """Reads evolutionary track data from the given file."""
    return ascii.read(f"{path}/{mass}M.track.eep")

def validate_input(min_val, max_val, prompt, command, var_type):
    """Prompts user for valid age range."""
    if command:
        command_var=command[var_type]
        if (not isfloat(command_var) and not isint(command_var)) and (min_val <= float(command_var) <= max_val):
            raise ValueError(f"Invalid {var_type} in: {command}")
        value=command_var
    else:
        value=input(prompt)
        while (not isfloat(value) and not isint(value)) and (min_val <= float(value) <= max_val):
            error_msg=f"Not a valid input. {prompt}: "
            value=input(error_msg)
    
    return float(value)

def interpolate_between_tracks(padded_mass_choice, lower_bound_padded, upper_bound_padded, path, step_size=0.001):
    """Interpolates between two sets of evolutionary track data."""
    df_lower=read_track_data(path, lower_bound_padded)
    age_lower=df_lower['col1']
    lum_lower=df_lower['col7']
    temp_lower=df_lower['col12']
    
    df_upper=read_track_data(path, upper_bound_padded)
    age_upper=df_upper['col1']
    lum_upper=df_upper['col7']
    temp_upper=df_upper['col12']

    target_step=0
    total_steps=0
    mass_choice_float=float(padded_mass_choice)/100
    lower_mass_bound_float=float(lower_bound_padded)/100
    upper_mass_bound_float=float(upper_bound_padded)/100
    while lower_mass_bound_float+total_steps < upper_mass_bound_float:
        if lower_mass_bound_float+target_step < mass_choice_float:
            # keep track of number of steps to user selected mass
            target_step+=step_size
        # keeps track of number of steps to upper bound
        total_steps+=step_size
    target_idx=math.ceil(target_step/step_size)-1 # -1 b/c 0 indexed
    num_steps=math.ceil(total_steps/step_size) # no need to subtract because ends when equal to upper mass bound
    print(f"WARNING: num_steps: {num_steps}; target_idx: {target_idx}.")
    temp_interp = [np.linspace(tl, th, num_steps)[target_idx] for tl, th in zip(temp_lower, temp_upper)]
    lum_interp = [np.linspace(ll, lh, num_steps)[target_idx] for ll, lh in zip(lum_lower, lum_upper)]
    age_interp = [np.linspace(al, ah, num_steps)[target_idx] for al, ah in zip(age_lower, age_upper)]
    return temp_interp, lum_interp, age_interp

def get_track_data(padded_mass_choice, file_ints_str, path, age_constraints, command):
    """returns temperature and luminosity data for a given stellar mass evolutionary track"""
    # Check if min_interp_dec exists in files
    if padded_mass_choice in file_ints_str:
        print(f"WARNING: Interpolation not needed for selected mass as file is available: {padded_mass_choice}M.track.eep")
        df = read_track_data(path, padded_mass_choice)
        age_arr=df['col1'] # AGE in years
        
        lum_arr=df['col7'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity
        temp_arr=df['col12'] # LOG EFFECTIVE TEMPERATURE in Kelvin 
        
    else: # need to interpolate for mass 
        print(f"WARNING: File not found, generating interpolated evolutionary track for the selected stellar mass.")
        # get the stellar mass files that upper bound and lower bound the user's selected mass choice 
        i=0
        while i < len(file_ints_str) and float(file_ints_str[i])/100 < float(padded_mass_choice)/100:
            i+=1
        i-=1
        temp_arr,lum_arr,age_arr= interpolate_between_tracks(padded_mass_choice, lower_bound_padded=file_ints_str[i], upper_bound_padded=file_ints_str[i+1], path=path)

    # only use the lower mass data to set the age range as they live longer than high mass stars
    if np.sum(age_constraints) == 0: 
        min_age, max_age = min(age_arr), max(age_arr)
        user_age_min = validate_input(
                                    min_age,
                                    max_age,
                                    f"Enter minimum age (between {min_age} and {max_age} [years]): ",
                                    command=command,
                                    var_type='min_age_years'
                                    )
    
        user_age_max = validate_input(
                                    user_age_min,
                                    max_age,
                                    f"Enter maximum age (between {user_age_min} and {max_age} [years]): ",
                                    command=command,
                                    var_type='max_age_years'
                                    )
    elif len(age_constraints) == 2: 
        user_age_min=age_constraints[0]
        user_age_max=age_constraints[-1]
        
    else: 
        raise ValueError(f"age_constraints not an array of length two: {age_constraints}")
    
    # TODO: could build interpolator using the entire evolutionary track and then just evaluate at the user specified age range
    # but this would be more computationally expensive than just filtering the arrays 
    age_arr, temp_arr, lum_arr = np.array(age_arr), np.array(temp_arr), np.array(lum_arr)
    mask = np.where((age_arr >= user_age_min) & (age_arr <= user_age_max))
    temp_arr = temp_arr[mask]
    lum_arr = lum_arr[mask]
    age_arr = age_arr[mask]
    return temp_arr, lum_arr, age_arr, user_age_min, user_age_max

def plot_eep(fig, ax, interactive, command={}, linestyle='-', debug=False):
    """Main function to plot evolutionary track with EEP interpolation."""
    if interactive:
        path = input("Input name of untarred evolutionary track file: ")
    else:
        path=command['path_to_untarred_ET']
    l_files = os.listdir(path)
    arr_clean = file_cleanup(l_files)
    
    # Sort the mass files in increasing order and present to user
    v_selec = sorted(arr_clean)
    file_ints_str = [file[:5] for file in v_selec]
    minimum_available_mass=float(min(file_ints_str))/100
    maximum_available_mass=float(max(file_ints_str))/100

    if debug:
        for i, file in enumerate(v_selec):
            print(f"{file} Solar Masses: {float(file_ints_str[i]) / 100}")
    
    # Get user input for interpolation bounds
    print("INSTRUCTIONS: Input mass. Ex: For 255.05 solar masses, enter 255.05")
    
    min_interp_dec = validate_input(
        minimum_available_mass, 
        maximum_available_mass,
        f"Input mass for lower bound evolutionary track curve (range {minimum_available_mass} - {maximum_available_mass} "+r"[M_sun]): ",
        command=command,
        var_type='lower_bound_dynamical_mass_solar_mass'
    )
    
    max_interp_dec = validate_input(
        min_interp_dec, 
        maximum_available_mass,
        f"Input mass for maximum bound evolutionary track curve (range {min_interp_dec} - {maximum_available_mass} "+r"[M_sun]): ",
        command=command,
        var_type='upper_bound_dynamical_mass_solar_mass'
    )
    
    padded_min_interp_dec=f"{min_interp_dec*100:0>5}"
    padded_max_interp_dec=f"{max_interp_dec*100:0>5}"
    
    min_temp_arr, min_lum_arr, min_age_arr, user_age_min, user_age_max=get_track_data(padded_min_interp_dec,file_ints_str,path,age_constraints=[0,0], command=command)
    max_temp_arr, max_lum_arr, max_age_arr,  _ , _ =get_track_data(padded_max_interp_dec,file_ints_str,path,age_constraints=[user_age_min, user_age_max], command=command)

    # use age arrays to define new age grid
    common_age_grid = np.linspace(max(min_age_arr[0], max_age_arr[0]), min(min_age_arr[-1], max_age_arr[-1]), num=500)

    # this partitions our evolutionary tracks and fits cubic splines to each parition
    # essentially it creates a piecewise function to model the evolutionary track allowing us to take a discrete 
    # evolutionary track and turn it into a continous function
    Akima_min_temp = Akima1DInterpolator(min_age_arr, min_temp_arr)
    Akima_min_lum = Akima1DInterpolator(min_age_arr, min_lum_arr)
    Akima_max_temp = Akima1DInterpolator(max_age_arr, max_temp_arr)
    Akima_max_lum = Akima1DInterpolator(max_age_arr, max_lum_arr)

    T1_new, L1_new = Akima_min_temp(common_age_grid), Akima_min_lum(common_age_grid)
    T2_new, L2_new = Akima_max_temp(common_age_grid), Akima_max_lum(common_age_grid)

    lower_bound_label = command['lower_bound_dynamical_mass_label']
    upper_bound_label = command['upper_bound_dynamical_mass_label']
    ax.plot(T1_new, L1_new, color='tab:blue', label=lower_bound_label)
    ax.plot(T2_new, L2_new, color='tab:orange', label=upper_bound_label)
    
    # Draw equal-age connecting lines
    previous_age = common_age_grid[0]
    previous_idx = 0 
    for i, age in enumerate(common_age_grid):
        plt.plot([T1_new[i], T2_new[i]], [L1_new[i], L2_new[i]], 'k--', alpha=0.4)
        # highlight every 700,000 years
        if common_age_grid[i] - previous_age >= 700_000 and i - previous_idx >= 3:
            ax.text((T1_new[i]+T2_new[i])/2, (L1_new[i]+L2_new[i])/2, f"{common_age_grid[i]:.1e} years", fontsize=8, color='purple', rotation=45)
            plt.plot([T1_new[i], T2_new[i]], [L1_new[i], L2_new[i]], 'k--', alpha=0.8, color='purple')
            previous_age = common_age_grid[i]
            previous_idx = i
    return fig, ax

# TODO -- make this interpolate ... though i think this will be removed now that we're using the akima interpolator
def plot_iso(fig, ax, interactive, command={}, linestyle='-', debug=False):
    print("Isocrhone: same age, different masses ")
    if interactive:
        path=input("Input name of untarred isochrone file: ")
    else:
        path=command['path_to_iso']

    vcritval_start_idx=path.rfind('vvcrit')+len('vvcrit')
    vcritval=path[vcritval_start_idx:vcritval_start_idx+3]
    print(vcritval)

    if interactive:
        print()
        print("[Fe/H] = -4.00 \t [Fe/H] = -3.50 \t [Fe/H] = -3.00")
        print("[Fe/H] = -2.50 \t [Fe/H] = -2.00 \t [Fe/H] = -1.75")
        print("[Fe/H] = -1.50 \t [Fe/H] = -1.25 \t [Fe/H] = -1.00")
        print("[Fe/H] = -0.75 \t [Fe/H] = -0.50 \t [Fe/H] = -0.25")
        print("[Fe/H] = +0.00 \t [Fe/H] = +0.25 \t [Fe/H] = +0.50")
        prompt="Choose [Fe/H]: "
        user_metallicity = int(input(prompt))
        valid = {
            '-4.00',
            '-3.50',
            '-3.00',
            '-2.50',
            '-2.00',
            '-1.75',
            '-1.50',
            '-1.25',
            '-1.00',
            '-0.75',
            '-0.50',
            '-0.25',
            '+0.00',
            '+0.25',
            '+0.50',
            }

        while user_metallicity not in valid:
            print(f"Invalid choice. ")
            user_metallicity = int(input(prompt))
    else:
        user_metallicity=command['[Fe/H]']

    prefix='p' if user_metallicity[0]=='+' else 'm'
    user_selected_file=f"{path}/MIST_v1.2_feh_{prefix}{user_metallicity[1:]}_afe_p0.0_vvcrit{vcritval}_UBVRIplus.iso.cmd"
    print(f"Selected file: {user_selected_file}")
    
    data = ascii.read(user_selected_file)
    logL = np.array(data['col7']) # log(L) [L_sun]?
    logTEFF = np.array(data['col5']) # log(T_EFF) [K]?
    MASS = np.array(data['col3']) # initil mass [M_sun]?
    logAGE = np.array(data['col2']) # log10(ISOCHRONE AGE) [years]
    AGE_yrs=10**logAGE

    min_age=min(AGE_yrs)
    max_age=max(AGE_yrs)
    user_age_min = validate_input(
                                min_age,
                                max_age,
                                f"Enter minimum age for isochrone (between {min_age} and {max_age} [years]): ",
                                command=command,
                                var_type='min_age_years'
                                )

    user_age_max = validate_input(
                                user_age_min,
                                max_age,
                                f"Enter maximum age for isochrone (between {user_age_min} and {max_age} [years]): ",
                                command=command,
                                var_type='max_age_years'
                                )   
    
    mask = np.where((AGE_yrs>user_age_min)&(AGE_yrs<user_age_max))
    label=label = input("Input label for isochrone curve: ") if interactive else command['label']
    ax.plot(logTEFF[mask], logL[mask], color = 'red', linestyle = linestyle, label=label)
    return fig, ax    