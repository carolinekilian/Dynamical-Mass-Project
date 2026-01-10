import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastnumbers import isfloat, isint
from astropy.io import ascii
from scipy.interpolate import Akima1DInterpolator, CubicSpline
import matplotlib as mpl
import re 
from sklearn.isotonic import IsotonicRegression


class ValidationTools: 
    
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

class ConversionTools:

    convert_to_log = lambda x: np.log10(x) if x > 0 else 0 
   # source: https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
    uncertainty_prop = lambda uncert, data_point: uncert/(np.log(10)*data_point)

class PlottingTools:
    mpl.rcParams['text.usetex'] = True

    def plot_point(fig, ax, color,command={}):
        """
        Function to plot a point with error bars on the HR diagram.
        """

        x_point_r = ValidationTools.get_valid_input(
            "Please provide the log effective temperature [Kelvin]: Log(T_eff) = ",
            "Invalid input. Please provide a valid number for Log(T_eff): ",
            command,
            'temperature_kelvin'
        )

        x_point_err_r = ValidationTools.get_valid_input(
            "Please provide the log error for effective temperature [Kelvin]: ",
            "Invalid input. Please provide a valid number for the error: ",
            command,
            'temperature_kelvin_err'
        )
        
        y_point_r = ValidationTools.get_valid_input(
            "Please provide the log bolometric luminosity of the star: Log(L) = ",
            "Invalid input. Please provide a valid number for Log(L): ",
            command,
            'luminosity_solar_lum'
        )
        y_point_err_r = ValidationTools.get_valid_input(
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
        x_point = ConversionTools.convert_to_log(x_point_r)  
        x_point_err = ConversionTools.uncertainty_prop(x_point_err_r, x_point_r) if x_point_err_r !=0 else 0
        print(f"x_point (log scale): {x_point} ± {x_point_err}")
        y_point =ConversionTools.convert_to_log(y_point_r)
        y_point_err = ConversionTools.uncertainty_prop(y_point_err_r, y_point_r) if y_point_err_r !=0 else 0
        print(f"y_point (log scale): {y_point} ± {y_point_err}")

        ax.scatter([x_point], [y_point], color=color)
        ax.errorbar([x_point], [y_point], xerr=[x_point_err], yerr=[y_point_err], label=name, color=color, fmt='o')
        return fig, ax

    def get_unique_color(available_colors, used_colors):
        if not available_colors:
            raise ValueError("No more unique colors available.")
        
        color = available_colors.pop(0)
        used_colors.add(color)

        return color, available_colors
    
    def plot_format(fig, ax, title, fontsize=30):

        plt.title(title, fontsize=fontsize)
        plt.xlabel(r'log ($T_{eff}$) [K]', fontsize=fontsize)
        plt.ylabel(r'log ($L_{bol}/L_{\odot}$)', fontsize=fontsize)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend()
        ax.legend(
        fontsize=20,      # Set the font size (can be int/float or string like 'large')
        markerscale=2,  # Scale the markers in the legend (1.5 makes them 50% larger)
        )
        plt.gca().invert_xaxis()
        plt.grid()
        plt.show()

    def plot_line_of_constant_temperature(fig, ax, y_limits, command_temp):
        min_temp=command_temp['min_temp_kelvin']
        max_temp=command_temp['max_temp_kelvin']
        label=command_temp['label']

        #convert to log scale
        min_temp=ConversionTools.convert_to_log(min_temp)
        max_temp=ConversionTools.convert_to_log(max_temp)

        # get min and max of y axis
        y_min, y_max = y_limits
        # fill in between min and max temperature
        ax.fill_betweenx([y_min, y_max], min_temp, max_temp, color='lightgray', alpha=0.5, label=label)
        return fig, ax

    def plot_line_of_constant_luminosity(fig, ax, x_limits, command_lum):
        min_lum=ConversionTools.convert_to_log(command_lum['min_lum_solar_lum'])
        max_lum=ConversionTools.convert_to_log(command_lum['max_lum_solar_lum'])
        label=command_lum['label']

        # get min and max of x axis
        x_min, x_max = x_limits
        # fill in between min and max luminosity
        ax.fill_between([x_min, x_max], min_lum, max_lum, color='lightgray', alpha=0.5, label=label)
        return fig, ax

class InterpolatorTools:

    def make_strictly_monotonic(x, increasing=True, eps=1e-6):
        ir = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
        y = ir.fit_transform(np.arange(len(x)), x)

        # enforce strictness
        for i in range(1, len(y)):
            if increasing:
                if y[i] <= y[i-1]:
                    y[i] = y[i-1] + eps
            else:
                if y[i] >= y[i-1]:
                    y[i] = y[i-1] - eps

        return y
    
    def find_monotonic_segments(age):
        """Find segments where the age is monotonically increasing or decreasing."""
        segments = []
        start_idx = 0
        age = InterpolatorTools.make_strictly_monotonic(age, increasing=True, eps=1e-6)
        # Determine sign of derivative
        dT = np.diff(age)
        sign_prev = np.sign(dT[0])
        for i in range(1, len(dT)):
            sign_now = np.sign(dT[i])
           
            if sign_now != sign_prev:
                segments.append(slice(start_idx, i + 1))
                start_idx = i

            sign_prev = sign_now
        segments.append(slice(start_idx, len(age)))
        return segments, age

    def make_piecewise_interpolator(age, values, method='cubic'):
        """Create piecewise interpolators for segments of monotonic data

            basically I initially tried fitting a spline to the entire evolutionary track
            but it failed because the temperature and luminosity are not monotonic functions of age
            so I broke the data into segments where the function is monotonic and fit splines to each segment
        """
        segments, age = InterpolatorTools.find_monotonic_segments(age)
        interpolators = []
        for seg in segments:
            t_seg, v_seg = age[seg], values[seg]
            # Need at least 3 points for cubic spline or Akima
            if len(t_seg) < 3:
                continue
            if method == 'cubic':
                interpolators.append(CubicSpline(t_seg, v_seg))
            elif method == 'akima':
                interpolators.append(Akima1DInterpolator(t_seg, v_seg))
            else:
                raise ValueError("Method must be 'cubic' or 'akima'")
        segment_bounds = [(age[s.start], age[s.stop - 1]) for s in segments]
        return segment_bounds, interpolators

    def piecewise_eval(age_query, segment_bounds, interpolators):
        result = np.empty_like(age_query)
        for i, t in enumerate(age_query):
            found = False
            for (tmin, tmax), interp in zip(segment_bounds, interpolators):
                if tmin <= t <= tmax:
                    result[i] = interp(t)
                    found = True
                    break
            if not found:
                # if not in any segment, use nearest endpoint
                if t < segment_bounds[0][0]:
                    result[i] = interpolators[0](t)
                else:
                    result[i] = interpolators[-1](t)
        return result
    
class ProcessingTools:
    def file_cleanup(files):
        """Cleans up list of files by removing README and system files."""
        exclude = {'README_tables.pdf', 'README_overview.pdf', '.DS_Store', 'BHAC15_tracks+structure.txt'}
        return [f for f in files if f not in exclude and f[-7:]!='ADD.DAT' and f[-7:]!='.HB.DAT']
    
    def read_track_data(directory, mass, source, sample_file):
        """Reads evolutionary track data from the given file."""
        if source=='MIST':
            # for parameter descriptions refer to the README files in the respective track directory 
            mass=f"{int(mass)*100:0>5}"
            df=ascii.read(f"{directory}/{mass}M.track.eep")
            age=df['col1'] # AGE in years
            lum=df['col7'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity
            temp=df['col12'] # LOG EFFECTIVE TEMPERATURE in Kelvin

        elif source=='BHAC':
            # for parameter descriptions refer to BHAC15_tracks+structure.txt in the respective track directory
            betweenMandp,afterp=str(mass).split('.')
            df=pd.read_csv(f"{directory}/BHAC15-M{betweenMandp}p{afterp:0<3}.txt", sep='\s+')
            df_new=df.drop(['Rrad','k2conv','k2rad'],axis=1)
            df_new.columns=['Mass (solar)','log t (yr)','Teff (K)','log(L/L_sun)','log g','R/Rs','Log(Li/Li0)','log Tc','log ROc','Mrad','Rrad','k2conv','k2rad']
            age=10**df_new['log t (yr)'] # AGE in years 
            lum=df_new['log(L/L_sun)'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity 
            temp=np.log10(df_new['Teff (K)']) # LOG EFFECTIVE TEMPERATURE in Kelvin 

        elif source=='Feiden Magnetic':
            ProcessingTools.rename_Feiden_Magnetic_data(directory)

            columns = [
                "model_number", "shells", "age", "logL_Lsun", "logR_Rsun",
                "log_g", "log_Teff", "Mconv_core", "Mconv_env", "Rconv_env"
            ]

            df = pd.read_csv(
                f"{directory}/m{int(mass*1000):0>4}_GS98_p000_p0_y28_mlt1.884.ntrk",
                comment='#',
                sep='\s+',
                header=None,
                usecols=range(len(columns)),  
                engine='python'
            )
            df.columns=columns # to check - assumed age was in MYRs
            age=df['age']*10**9 # AGE in years 
            lum=df['logL_Lsun'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity 
            temp=df['log_Teff'] # LOG EFFECTIVE TEMPERATURE in Kelvin

        elif source=='Feiden Non-Magnetic':

            columns = [
                        "Age (yrs)", "Log T", "Log g", "Log L", "Log R", "Y_core", "Z_core", "(Z/X)_surf",  "A(Li)",
                        "Li_H",  "k2", "B_tach", "u_conv", "t_conv"
                ]

            df = pd.read_csv(
                f"{directory}/m{int(mass*1000):0>4}_GS98_p000_p0_y28_mlt1.884.trk",
                comment='#',
                sep='\s+',
                header=None,
                usecols=range(len(columns)),
                engine='python'
            )
        
            df.columns = columns # to check - assumed log L being in solar luminsoities and temp in Kelvin 
            age=df["Age (yrs)"] # AGE in years 
            lum=df['Log L'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity  
            temp=df['Log T'] # LOG EFFECTIVE TEMPERATURE in Kelvin

        elif source=='PARSEC1.2S':
            ProcessingTools.rename_PARSEC1p2S_data(directory)
            file_prefix, _=sample_file.split('M')
            integer_mass=f"{str(mass).split('.')[0]:0>3}"
            fractional_mass=f"{str(mass).split('.')[1]:0<3}"
            df=pd.read_csv(f"{directory}/{file_prefix}M{integer_mass}.{fractional_mass}.DAT", sep='\s+')
            age=df['AGE'] # AGE in years 
            temp=df['LOG_TE'] # LOG EFFECTIVE TEMPERATURE in Kelvin
            lum=df['LOG_L'] # LOG BOLOMETRIC LUMINOSITY in solar luminosity  
            
        return age, lum, temp
    
    def rename_PARSEC1p2S_data(directory):
        """standardizes PARSEC file naming convention by padding whole numbers"""
        for file in os.listdir(directory):
            beforeM, afterM=file.split('M')
            before_decimal,after_decimal=afterM.split('.', maxsplit=1)
            old_path=os.path.join(directory,file)
            new_file_name=os.path.join(directory,f"{beforeM}M{before_decimal:0>3}.{after_decimal}")
            os.rename(old_path,new_file_name)
            #print(f"Renamed {old_path} to {new_file_name}")
        return 

    def rename_Feiden_Magnetic_data(directory):
        """standardizes Feiden Magnetic file naming convention by removing _mag*.ntrk and replacing with .ntrk"""
        for filename in os.listdir(directory):
            if filename.endswith(".ntrk"):
                new_name = re.sub(r'_mag[^.]*\.ntrk$', '.ntrk', filename)
                if new_name != filename:
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_name)
                    # print(f"Renaming: {filename} → {new_name}")
                    os.rename(old_path, new_path)
        return 

def interpolate_between_tracks(padded_mass_choice, lower_bound_padded, upper_bound_padded, directory, source, sample_file):
    """Interpolates between two sets of evolutionary track data."""    
    age_lower, lum_lower, temp_lower=ProcessingTools.read_track_data(directory, lower_bound_padded, source, sample_file)
    age_upper, lum_upper, temp_upper=ProcessingTools.read_track_data(directory, upper_bound_padded, source, sample_file)

    interpolator='akima'  # 'cubic' or 'akima'
    bounds_temp_lower, interps_temp_lower = InterpolatorTools.make_piecewise_interpolator(age_lower, temp_lower, method=interpolator)
    bounds_lum_lower, interps_lum_lower = InterpolatorTools.make_piecewise_interpolator(age_lower, lum_lower, method=interpolator)
    bounds_temp_upper, interps_temp_upper = InterpolatorTools.make_piecewise_interpolator(age_upper, temp_upper, method=interpolator)
    bounds_lum_upper, interps_lum_upper = InterpolatorTools.make_piecewise_interpolator(age_upper, lum_upper, method=interpolator)

    # Define a common age grid based on the overlapping age range
    common_age_grid = np.linspace(max(min(age_lower), min(age_upper)), min(max(age_lower), max(age_upper)), num=1_000)

    # Evaluate the interpolators on the common age grid
    T_lower_new = InterpolatorTools.piecewise_eval(common_age_grid, bounds_temp_lower, interps_temp_lower)
    L_lower_new = InterpolatorTools.piecewise_eval(common_age_grid, bounds_lum_lower, interps_lum_lower)
    T_upper_new = InterpolatorTools.piecewise_eval(common_age_grid, bounds_temp_upper, interps_temp_upper)
    L_upper_new = InterpolatorTools.piecewise_eval(common_age_grid, bounds_lum_upper, interps_lum_upper)

    # Linear interpolation between the two tracks based on mass
    mass_fraction = (padded_mass_choice - lower_bound_padded) / (upper_bound_padded - lower_bound_padded)
    temp_interp = T_lower_new + mass_fraction * (T_upper_new - T_lower_new)
    lum_interp = L_lower_new + mass_fraction * (L_upper_new - L_lower_new)
    age_interp = common_age_grid

    return temp_interp, lum_interp, age_interp

def get_track_data(padded_mass_choice, file_ints_str, directory, age_constraints, command, sample_file):
    """returns temperature and luminosity data for a given stellar mass evolutionary track"""
    source=command['source']
    # Check if min_interp_dec exists in files
    if padded_mass_choice in file_ints_str:
        print(f"WARNING: Interpolation not needed for selected mass as file is available: {int(padded_mass_choice*100):0>5}M.track.eep")
        age_arr,lum_arr,temp_arr=ProcessingTools.read_track_data(directory, padded_mass_choice, source, sample_file)
        
    else: # need to interpolate for mass 
        print(f"WARNING: File not found, generating interpolated evolutionary track for the selected stellar mass.")
        # get the stellar mass files that upper bound and lower bound the user's selected mass choice 
        i=0
        while i < len(file_ints_str) and file_ints_str[i] < padded_mass_choice:
            i+=1

        if i == len(file_ints_str): 
            i-=2
        else: 
            i-=1 
        # TODO: should generalize code so that tells user "mass is out of range of dataset" 
        temp_arr,lum_arr,age_arr= interpolate_between_tracks(padded_mass_choice, lower_bound_padded=file_ints_str[i], upper_bound_padded=file_ints_str[i+1], directory=directory, source=source, sample_file=sample_file)

    # only use the lower mass data to set the age range as they live longer than high mass stars
    if np.sum(age_constraints) == 0: 
        min_age, max_age = min(age_arr), max(age_arr)
        user_age_min = ValidationTools.validate_input(
                                    min_age,
                                    max_age,
                                    f"Enter minimum age (between {min_age} and {max_age} [years]): ",
                                    command=command,
                                    var_type='min_age_years'
                                    )
    
        user_age_max = ValidationTools.validate_input(
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
    
    original_age_arr, original_temp_arr, original_lum_arr = np.array(age_arr), np.array(temp_arr), np.array(lum_arr)
    # fit a cubic spline to the extracted/interpolated data 
    bounds_temp, temp_interpolator = InterpolatorTools.make_piecewise_interpolator(original_age_arr, original_temp_arr, method='cubic')
    bounds_lum, lum_interpolator = InterpolatorTools.make_piecewise_interpolator(original_age_arr, original_lum_arr, method='cubic')

    # build an interpolator using the extracted/interpolated data WITHIN the user defined age range
    user_age_arr = np.linspace(user_age_min, user_age_max, num=1_000)
    temp_arr = InterpolatorTools.piecewise_eval(user_age_arr, bounds_temp, temp_interpolator)
    lum_arr = InterpolatorTools.piecewise_eval(user_age_arr, bounds_lum, lum_interpolator)
    
    return temp_arr, lum_arr, age_arr, user_age_min, user_age_max

def plot_eep(fig, ax, interactive, color_map, command={}, source='MIST', linestyle='-', debug=False):
    """Main function to plot evolutionary track with EEP interpolation."""
    if interactive:
        directory = input("Input name of untarred evolutionary track file: ")
    else:
        directory=command['path_to_untarred_ET']

    l_files = os.listdir(directory)
    source=command['source']
    only_data_files = sorted(ProcessingTools.file_cleanup(l_files))
    if source=='MIST' or source=='BHAC':
        if source=='MIST':
            file_ints_str = [float(file[:5])/100 for file in only_data_files]
        elif source=='BHAC':
            file_ints_str = [float(file.split('M')[1].split('p')[0]+'.'+file.split('p')[1].split('.txt')[0]) for file in only_data_files]
    elif source=='PARSEC1.2S':
            file_ints_str = [float(file.split('M')[1].split('.DAT', maxsplit=1)[0]) for file in only_data_files]
    elif source=='Feiden-Magnetic' or 'Feiden-Non-Magnetic':
        file_ints_str = [float(file[1:5])/1000 for file in only_data_files]
    
    minimum_available_mass=float(min(file_ints_str))
    maximum_available_mass=float(max(file_ints_str))
    # Get user input for interpolation bounds
    print("INSTRUCTIONS: Input mass. Ex: For 255.05 solar masses, enter 255.05")
    
    min_interp_dec = ValidationTools.validate_input(
        minimum_available_mass, 
        maximum_available_mass,
        f"Input mass for lower bound evolutionary track curve (range {minimum_available_mass} - {maximum_available_mass} "+r"[M_sun]): ",
        command=command,
        var_type='lower_bound_dynamical_mass_solar_mass'
    )
    
    max_interp_dec = ValidationTools.validate_input(
        min_interp_dec, 
        maximum_available_mass,
        f"Input mass for maximum bound evolutionary track curve (range {min_interp_dec} - {maximum_available_mass} "+r"[M_sun]): ",
        command=command,
        var_type='upper_bound_dynamical_mass_solar_mass'
    )
    
    padded_min_interp_dec=min_interp_dec
    padded_max_interp_dec=max_interp_dec
    min_temp_arr, min_lum_arr, min_age_arr, user_age_min, user_age_max=get_track_data(padded_min_interp_dec,file_ints_str,directory,age_constraints=[0,0], command=command, sample_file=only_data_files[0])
    max_temp_arr, max_lum_arr, max_age_arr,  _ , _ =get_track_data(padded_max_interp_dec,file_ints_str,directory,age_constraints=[user_age_min, user_age_max], command=command, sample_file=only_data_files[0])
    
    lower_bound_label = f" {source}"
    ax.plot(min_temp_arr, min_lum_arr, lw=3, color=color_map[command['source']],label=lower_bound_label)
    ax.plot(max_temp_arr, max_lum_arr, lw=3, color=color_map[command['source']])
    
    # Draw equal-age connecting lines
    previous_age = min_age_arr[0]
    previous_idx = 0 
    
    for i, age in enumerate(min_age_arr):
        
        plt.plot([min_temp_arr[i], max_temp_arr[i]], [min_lum_arr[i], max_lum_arr[i]], '--', alpha=0.8, color=color_map[command['source']])
        # highlight every 700,000 years
        if age - previous_age >= 700_000 and i - previous_idx >= 3:
            #ax.text((T1_new[i]+T2_new[i])/2, (L1_new[i]+L2_new[i])/2, f"{common_age_grid[i]:.1e} years", fontsize=8, color='black', rotation=45)
            #plt.plot([T1_new[i], T2_new[i]], [L1_new[i], L2_new[i]], '--', alpha=0.8, color=color_map[command['source']])
            previous_age = age
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
    user_age_min = ValidationTools.validate_input(
                                min_age,
                                max_age,
                                f"Enter minimum age for isochrone (between {min_age} and {max_age} [years]): ",
                                command=command,
                                var_type='min_age_years'
                                )

    user_age_max = ValidationTools.validate_input(
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