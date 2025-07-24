from files.download_eep import download_eep
from files.download_iso import download_isochrone
from files.isochrone import plt_iso
from files.evolutionary_track import plot_eep 
from fastnumbers import isint, isfloat
import matplotlib.pyplot as plt
import os


# /Users/joshuagrajales/Downloads/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS.txz
# /Users/joshuagrajales/Desktop/astrre/final_plot
def run():

    ''' cwd = input("Enter absolute path to current working directory : ")
    while os.path.exists(cwd) != True:
        cwd = input("Not a valid directroy. Enter absolute path to current working directory : ")
 '''

    print("1. Download evolutionary track files")
    print("2. Download ischorone files")
    print("3. Input data for evolutionary track curves ")
    print("4. Input data for isochrone curves")
    print("5. Plot all")
    print("6. Quit")

    decision = input("Enter a digit with the task associated above. ")
    while decision != '6':
        if isint(decision) != True or int(decision) > 6 or int(decision) < 1:
            decision = int(input("Not a valid input. Please enter a digit with the task associated above. "))
        elif int(decision) == 1:
            download_eep()
            print("1. Download evolutionary track files")
            print("2. Download ischorone files")
            print("3. Input data for evolutionary track curves ")
            print("4. Input data for isochrone curves")
            print("5. Plot all")
            print("6. Quit")
            decision = input("Enter a digit with the task associated above. ")
        elif int(decision) == 2:
            download_isochrone()
            print("1. Download evolutionary track files")
            print("2. Download ischorone files")
            print("3. Input data for evolutionary track curves ")
            print("4. Input data for isochrone curves")
            print("5. Plot all")
            print("6. Quit")
            decision = input("Enter a digit with the task associated above. ")
        elif int(decision) == 3:
            # /Users/joshuagrajales/Downloads/MIST_v1.2_feh_m0.50_afe_p0.0_vvcrit0.4_EEPS.txz
            plot_eep()
            print("1. Download evolutionary track files")
            print("2. Download ischorone files")
            print("3. Input data for evolutionary track curves ")
            print("4. Input data for isochrone curves")
            print("5. Plot all")
            print("6. Quit")
            decision = input("Enter a digit with the task associated above. ")
        elif int(decision) == 4:
            # /Users/joshuagrajales/Downloads/MIST_v1.2_vvcrit0.4_UBVRIplus.txz
            plt_iso()
            print("1. Download evolutionary track files")
            print("2. Download ischorone files")
            print("3. Input data for evolutionary track curves ")
            print("4. Input data for isochrone curves")
            print("5. Plot all")
            print("6. Quit")
            decision = input("Enter a digit with the task associated above. ")
        elif int(decision) == 5:
            point = input("Would you like to plot a point? (yes or no) ")
            colors = ['red','dodgerblue','purple','green','hotpink','darkorange','slategrey']
            index = -1
            while point == 'yes':
                index+=1

                # ask for x point 
                x_point = input("Please provide the effective temperature in Kelvin: Log(T_eff) =  ")
                while isint(x_point) == False and isfloat(x_point) == False:
                    x_point = input("Invalid input. Please provide the effective temperature in Kelvin: Log(T_eff) =  ")
                x_point = [float(x_point)]
                x_point_err = input("Please provide the error for effective temperature in Kelvin: Log(T_eff) =  ")
                while isint(x_point_err) == False and isfloat(x_point_err) == False:
                    x_point_err = input("Invalid input. Please provide the effective temperature in Kelvin: Log(T_eff) =  ")
                x_point_err = [float(x_point_err)]

                # ask for y point 
                y_point = input("Please provide the bolometric luminosity of the star in: Log(L) = ")
                while isint(y_point) == False and isfloat(y_point) == False:
                    y_point = input("Invalid input. Please provide the effective temperature in Kelvin: Log(T_eff) =  ")
                y_point = [float(y_point)]
                y_point_err = input("Please provide the error for bolometric luminsity in Log(L) =  ")
                while isint(y_point_err) == False and isfloat(y_point_err) == False:
                    y_point_err = input("Invalid input. Please provide the error for bolometric luminsity in Log(L) =  ")
                y_point_err = [float(y_point_err)]

                # name for graph 
                name = input("Please provide a name for this point: ")
                plt.scatter(x_point, y_point,c=colors[index])
                plt.errorbar(x_point, y_point, yerr= y_point_err, xerr=x_point_err, label = name,c=colors[index])
                point = input("Would you like to plot another point? (yes or no): ")
            

            title = input("Please input the title to the graph: ")
            xlabel = input("Enter x label: ")
            ylabel = input("Enter y label: ")
            plt.title(title, fontsize = 15)
            plt.xlabel(xlabel, fontsize = 10)
            plt.ylabel(ylabel, fontsize = 10)
            plt.legend()
            plt.gca().invert_xaxis()
            plt.grid()
            plt.show()

    
            return 

    return 

run()



