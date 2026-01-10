from evolutionary_track import *
from fastnumbers import isint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

color_map = {
        "BHAC": "cornflowerblue", 
        "PARSEC1.2S": "darkgreen",
        "Feiden Non-Magnetic": "goldenrod",
        "Feiden Magnetic": "orangered",
        "MIST": "darkorchid"
        }

def run(interactive, commands={}, color_map=color_map):
    # Colors for the data points in the HR diagram
    # Get a list of all allowed colors in matplotlib
    available_colors = list(mcolors.TABLEAU_COLORS)#['black', 'blue', 'red', 'orange', 'grey', 'purple', 'green']
    used_colors = set()
    fig, ax = plt.subplots()
    if interactive:
        while True:
            print("1. Input data for evolutionary track curves")
            print("2. Input data for isochrone curves")
            print("3. Plot all")
            decision = input("Enter a digit with the task associated above: ")

            if not isint(decision) or not (1 <= int(decision) <= 2):
                print("Not a valid input.")
                continue
            decision = int(decision)

            if decision == 1:
                fig, ax = plot_eep(fig, ax, interactive, color_map)
            elif decision == 2:
                fig, ax = plot_iso(fig, ax, interactive)
            elif decision == 3:
                fig, ax = plot_more = input("Would you like to plot a point? (yes or no): ").lower()
                while plot_more == 'yes':
                    color, available_colors = PlottingTools.get_unique_color(available_colors, used_colors)
                    fig, ax = PlottingTools.plot_point(fig, ax, color)
                    plot_more = input("Would you like to plot another point? (yes or no): ").lower()
                title = input("Please input the title for the graph: ")
                PlottingTools.plot_format(fig, ax, title=title)
                return
    else:
        # plot all your ETs
        for command_et in commands.get('1',[]):
            if command_et['source'] == 'BHAC' and command_et['lower_bound_dynamical_mass_solar_mass'] > 1.4:
                print("BHAC tracks above 1.4 solar masses are not available. Skipping this track.")
                continue 

            for command_lum in commands.get('4',[]):
                command_lum['source']=command_et['source']
            for command_pp in commands.get('3', {}).get('pp', []):
                command_pp['source']=command_et['source']

            print(f"Generating {command_pp['source']} track")
            fig, ax = plot_eep(fig, ax, interactive, color_map, command_et)
        for command_iso in commands.get('2',[]):
            fig, ax = plot_iso(fig, ax, interactive,command_iso)
        # plot all of your points
        for command_pp in commands.get('3', {}).get('pp', []):
            color, available_colors = PlottingTools.get_unique_color(available_colors, used_colors)
            fig, ax = PlottingTools.plot_point(fig, ax, color, command_pp)
        
        # plot constant temperature lines 
        y_limits = ax.get_ylim()
        for command_temp in commands.get('4', {}).get('temperature_lines', []):
            fig, ax = PlottingTools.plot_line_of_constant_temperature(fig, ax, y_limits, command_temp)
        # plot constant luminosity lines
        x_limits = ax.get_xlim()
        for command_lum in commands.get('4', {}).get('luminosity_lines', []):
            fig, ax = PlottingTools.plot_line_of_constant_luminosity(fig, ax, x_limits, command_lum)
        title=commands['title']
    
        PlottingTools.plot_format(fig, ax, title=title)
        return

if __name__ == "__main__":
    # TODO: update interactive mode so that it can highlight regions of constant temp and lum 
    # run(interactive=True, commands={})
    # all evolutionary tracks should be placed in the hrplot/ directory. Each file should be unzipped/untarred
    lower_mass = 10**(0.192-0.012)
    upper_mass = 10**(0.192+0.010)
    lower_age = 14e6
    upper_age = 18e6

    commands={
                    '1': [ # the dynamical mass outputs go here
                        {   'source': 'PARSEC1.2S',
                            'path_to_untarred_ET':'all_tracks_Pv1.2s/Z0.0001Y0.249',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },  
                        {   'source': 'Feiden Non-Magnetic',
                            'path_to_untarred_ET':'all_GS98_p000_p0_y28_mlt1.884 - Feiden Non-Magnetic',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },  
                        {   'source': 'Feiden Magnetic',  
                            'path_to_untarred_ET':'all__GS98_p000_p0_y28_mlt1.884_Beq - Feiden Magnetic',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },  
                        {   'source': 'MIST',  
                            'path_to_untarred_ET':'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        }   
                    ],  
                    # '2': [ # isochrones go here
                    #     {   
                    #         'path_to_iso':'MIST_v1.2_vvcrit0.0_UBVRIplus',
                    #         '[Fe/H]':'-0.25',
                    #         'min_age_years':6.4e6,
                    #         'max_age_years':7.9e6,
                    #         'label':'iso curve'
                    #     }   
                    # ],
                    '3': { # spectroscopic and photometric points go here
                        'pp': [
                            {   
                                'temperature_kelvin':7500,
                                'temperature_kelvin_err':100,
                                'luminosity_solar_lum':8.92,
                                'luminosity_solar_lum_err':0.09,
                                'name':'Matra et al. (2020)'
                            },  
                            # {
                            #     'temperature_kelvin':11250,
                            #     'temperature_kelvin_err':26,
                            #     'luminosity_solar_lum':24.4,
                            #     'luminosity_solar_lum_err':0.2,
                            #     'name':'Gaia Collaboration et al. (2023)'
                            # }, 
                            # {
                            #     'temperature_kelvin':8382,
                            #     'temperature_kelvin_err':99,
                            #     'luminosity_solar_lum':7.585,
                            #     'luminosity_solar_lum_err':0.017,
                            #     'name':'Zhang et al. (2023)'
                            # }, 
                        ],  
                    },  
                    '4': { # lines of constant temperature and luminosity go here
                        # 'temperature_lines':[
                        #     {   
                        #         'min_temp_kelvin':3500,
                        #         'max_temp_kelvin':3700,
                        #         'label':'Temp range 1'
                        #     },  
                        #     {   
                        #         'min_temp_kelvin':3800,
                        #         'max_temp_kelvin':4000,
                        #         'label':'Temp range 2'
                        #     }   
                        # ],
                        # 'luminosity_lines':[
                        #     {   
                        #         'min_lum_solar_lum':15,
                        #         'max_lum_solar_lum':17,
                        #         'label':'Lum range 1'
                        #     },  
                        #     {   
                        #         'min_lum_solar_lum':18,
                        #         'max_lum_solar_lum':20,
                        #         'label':'Lum range 2'
                        #     }   
                        # ] 
                    },  
                'title': f'HD131835: [{round(lower_mass, 3)}-{round(upper_mass, 3)}] '+r'$M_{\odot}$'
                }   
    # testing interactive
    run(interactive=False, commands=commands, color_map=color_map)