from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import os
'done'
s = Service("/usr/local/bin/chromedriver")
def download_isochrone():
    """
    downloads isochrone files used for constant mass curves in pltUBVRI.py
    """
    driver = webdriver.Chrome(service=s)
    driver.get("http://waps.cfa.harvard.edu/MIST/model_grids.html")
    driver.maximize_window()
    driver.find_element(By.LINK_TEXT, "UBV(RI)c + 2MASS JHKs + Kepler + Hipparcos + Tycho + Gaia (116MB)").click()
 
#download_isochrone()