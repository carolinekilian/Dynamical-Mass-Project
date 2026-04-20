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

            # for command_lum in commands.get('4',[]):
            #     command_lum['source']=command_et['source']
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
    star_name = 'HD 156623'
    lower_mass = 10**(0.248-0.023)
    upper_mass = 10**(0.248+0.023)
    lower_age = 10e6
    upper_age = 23e6

    sigfigs=4
    commands={
                    '1': [ # the dynamical mass outputs go here
                        {   'source': 'PARSEC1.2S', # read doc to select corresponding metallicity (close to Z=0.0)
                            'path_to_untarred_ET':'all_tracks_Pv1.2s/Z0.0001Y0.249',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },
                        {   'source': 'Feiden Non-Magnetic', # only available metallicity (close to Z=0.0)
                            'path_to_untarred_ET':'all_GS98_p000_p0_y28_mlt1.884 - Feiden Non-Magnetic',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },
                        {   'source': 'Feiden Magnetic', # only available metallicity (close to Z=0.0)
                            'path_to_untarred_ET':'all__GS98_p000_p0_y28_mlt1.884_Beq - Feiden Magnetic',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        },
                        {   'source': 'MIST', # (close to Z=0.0)
                            'path_to_untarred_ET':'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS',
                            'lower_bound_dynamical_mass_solar_mass': lower_mass,
                            'upper_bound_dynamical_mass_solar_mass': upper_mass,
                            'min_age_years': lower_age,
                            'max_age_years': upper_age,
                        }
                    ],
                    '3': { # spectroscopic and photometric points go here
                       'pp': [
                            {
                                'temperature_kelvin':8350,
                                'temperature_kelvin_err':0,
                                'luminosity_solar_lum':13.06,
                                'luminosity_solar_lum_err':1.80,
                                'name':'Moor et al. (2025)'
                            },
                            # {
                            #     'temperature_kelvin':9590,
                            #     'temperature_kelvin_err':27,
                            #     'luminosity_solar_lum':18.6,
                            #     'luminosity_solar_lum_err':0.2,
                            #     'name':'Kuchner et al. (2016)'
                            # },
                       ],
                    },
                    '4': { # lines of constant temperature and luminosity go here
                        # 'temperature_lines':[
                        #     {
                        #         'min_temp_kelvin':8700-210,
                        #         'max_temp_kelvin':8700+210,
                        #         'label':'Brennan et al. (2024)'
                        #     },
                            # {
                            #     'min_temp_kelvin':10174-524,
                            #     'max_temp_kelvin':10174+524,
                            #     'label':'Cataldi et al. (2023)'
                            # }
                        # ],
                        # 'luminosity_lines':[
                        #     {
                        #         'min_lum_solar_lum':10.352-0.015,
                        #         'max_lum_solar_lum':10.352+0.015,
                        #         'label':'Cataldi et al. (2023a)'
                        #     },
                        #     {
                        #         'min_lum_solar_lum':17.0,
                        #         'max_lum_solar_lum':17.3,
                        #         'label':'Matra et al. (2018)'
                        #     },
                        #     {
                        #         'min_lum_solar_lum':23.73,
                        #         'max_lum_solar_lum':23.81,
                        #         'label':'Cataldi et al. (2023a)'
                        #     }
                        # ]
                    },
                'title': f'{star_name}: [{lower_mass:.{sigfigs}f}-{upper_mass:.{sigfigs}f}] ' + r'$M_{\odot}$'
                }
    # testing interactive
    run(interactive=False, commands=commands, color_map=color_map)