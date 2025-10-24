from evolutionary_track import *
from fastnumbers import isint
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def run(interactive, commands={}):
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
                fig, ax = plot_eep(fig, ax, interactive)
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
            
            for command_lum in commands.get('4',[]):
                command_lum['source']=command_et['source']
            for command_pp in commands.get('3', {}).get('pp', []):
                command_pp['source']=command_et['source']

            fig, ax = plot_eep(fig, ax, interactive, command_et)
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
    commands=   {
                    '1': [ # the dynamical mass outputs go here 
                        {   'source': 'PARSEC1.2S',
                            'path_to_untarred_ET':'all_tracks_Pv1.2s/Z0.001Y0.25',                            
                            'lower_bound_dynamical_mass_solar_mass': 1.100,
                            'lower_bound_dynamical_mass_label': r'1.00 $M_{\odot}$',
                            'upper_bound_dynamical_mass_solar_mass': 1.180,
                            'upper_bound_dynamical_mass_label': r'1.105 $M_{\odot}$',
                            'min_age_years': 5e6,
                            'max_age_years': 15e7,
                        },
                        # {   'source': 'Feiden Non-Magnetic',
                        #     'path_to_untarred_ET':'all_GS98_p000_p0_y28_mlt1.884 - Feiden Non-Magnetic',
                        #     'lower_bound_dynamical_mass_solar_mass': 1.300,
                        #     'lower_bound_dynamical_mass_label': r'1.00 $M_{\odot}$',
                        #     'upper_bound_dynamical_mass_solar_mass': 1.380,
                        #     'upper_bound_dynamical_mass_label': r'1.105 $M_{\odot}$',
                        #     'min_age_years': 5e6,
                        #     'max_age_years': 15e7,
                        # },
                        # {   'source': 'Feiden Magnetic',
                        #     'path_to_untarred_ET':'all__GS98_p000_p0_y28_mlt1.884_Beq - Feiden Magnetic',
                        #     'lower_bound_dynamical_mass_solar_mass': 1.300,
                        #     'lower_bound_dynamical_mass_label': r'1.00 $M_{\odot}$',
                        #     'upper_bound_dynamical_mass_solar_mass': 1.380,
                        #     'upper_bound_dynamical_mass_label': r'1.105 $M_{\odot}$',
                        #     'min_age_years': 5e6,
                        #     'max_age_years': 15e7,
                        # },
                        # {   'source': 'BHAC',
                        #     'path_to_untarred_ET':'BHAC15',
                        #     'lower_bound_dynamical_mass_solar_mass': 1.100,
                        #     'lower_bound_dynamical_mass_label': r'1.00 $M_{\odot}$',
                        #     'upper_bound_dynamical_mass_solar_mass': 1.105,
                        #     'upper_bound_dynamical_mass_label': r'1.105 $M_{\odot}$',
                        #     'min_age_years': 5e6,
                        #     'max_age_years': 15e7,
                        # },
                        # {   'source': 'MIST',
                        #     'path_to_untarred_ET':'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_EEPS',
                        #     'lower_bound_dynamical_mass_solar_mass': 1.10,
                        #     'lower_bound_dynamical_mass_label': r'2.10 $M_{\odot}$',
                        #     'upper_bound_dynamical_mass_solar_mass': 1.18,
                        #     'upper_bound_dynamical_mass_label': r'2.15 $M_{\odot}$',
                        #     'min_age_years': 5e6,
                        #     'max_age_years': 15e6,
                        # }
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
                        # 'pp': [
                        #     {
                        #         'temperature_kelvin':3600,
                        #         'temperature_kelvin_err':300,
                        #         'luminosity_solar_lum':13,
                        #         'luminosity_solar_lum_err':2,
                        #         'name':'dummy1'
                        #     },
                        #     {
                        #         'temperature_kelvin':3700,
                        #         'temperature_kelvin_err':0,
                        #         'luminosity_solar_lum':20,
                        #         'luminosity_solar_lum_err':0,
                        #         'name':'dummy2'
                        #     }
                        # ],
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
                'title': 'test_title'
                }
    
    run(interactive=False, commands=commands)