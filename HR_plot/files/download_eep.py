from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

# location of chromedriver
s = Service("/usr/local/bin/chromedriver")
def download_eep():
    '''
    downloads appropriate evolutionary track file as specified by the user
    '''

    print()
    
    print("------------------------------------- INSTRUCTIONS -------------------------------------")
    print("Instructions: The following will download the Evolutionary Track file according to your specifications. Note the code will terminate once")
    print("download is initialized. When the evolutionary track file is finished downloading you must untar the file in the terminal.")
    #print("MAC users: 'tar -xvf <filename>' in the terminal.")
    #print("Windows users: '7z x filename'. tar in command prompt")

    print()
    print()
    
    print("------------------------------------- Download Evolutionary Track -------------------------------------")
    print("What evolutionary track would you like to download: ")
    print("A: v/vcrit = 0.4 ")
    print("B: v/vcrit = 0.0 ")
    
    # display options for v/vcrit
    user_input1 = input("Select A or B (case sensitive). ")
    while user_input1 != "B" and user_input1 != "A":
        print("Not a valid input please select A or B.")
        user_input1 = input("Select A or B (case sensitive). ")

    # display metallicity options
    print("What [Fe/H] would you like: ")
    print("1. [Fe/H] = -4.00 \t 2. [Fe/H] = -3.50 \t 3. [Fe/H] = -3.00")
    print("4. [Fe/H] = -2.50 \t 5. [Fe/H] = -2.00 \t 6. [Fe/H] = -1.75")
    print("7. [Fe/H] = -1.50 \t 8. [Fe/H] = -1.25 \t 9. [Fe/H] = -1.00")
    print("10. [Fe/H] = -0.75 \t 11. [Fe/H] = -0.50 \t 12. [Fe/H] = -0.25")
    print("13. [Fe/H] = -0.00 \t 14. [Fe/H] = +0.25 \t 15. [Fe/H] = +0.50")
    
    # metallicity logic checks
    user_input2 = int(input("Select a valid number index (1~15) for [Fe/H]: "))
    while type(user_input2) != int or user_input2 > 15 or user_input2 < 1:
        print("Not a valid input, try again.")
        user_input2 = int(input("Select a valid number index (1~15) for [Fe/H]: "))

    # the arrays below were indexed so that they match the html of the website
    # it was implemented such that the user input can be manipulated to correspond to an array index 
    # in the 'array' array. The 'array' array then holds the number that corresponds to the download link
    # in the html 
    if user_input1 == "A":
        array = [54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
        xpath_index = int(array[user_input2-1])
    elif user_input1 == "B":
        array = [69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]
        xpath_index = int(array[user_input2-1])

    if user_input1 == "A" and xpath_index < 69:

        # xpaths are up from 54 to 68 inclusive for 0.40 0.40 v/vcrit
        driver = webdriver.Chrome()
        driver.get("http://waps.cfa.harvard.edu/MIST/model_grids.html")
        driver.maximize_window()
        driver.find_element(By.XPATH,'//*[@id="content"]/article/p['+str(xpath_index)+']/a').click()

    elif user_input1 == "B" and xpath_index >= 69:

        #xpath is from 69 to 83 inclusive for 0.00 v/vcrit
        driver = webdriver.Chrome()
        driver.get("http://waps.cfa.harvard.edu/MIST/model_grids.html")
        driver.maximize_window()
        driver.find_element(By.XPATH,'//*[@id="content"]/article/p['+str(xpath_index)+']/a').click()




