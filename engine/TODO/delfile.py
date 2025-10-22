import os
from pathlib import Path
'''
obsdate = 0
while obsdate != 'Jan1' and obsdate != 'Jun21':
	obsdate = str(input('What is the observation date for HD32297?\n(abbreviate month and do not add spaces): '))
	filestart = './HD32297'+obsdate+'_'
	if obsdate != 'Jan1' and obsdate != 'Jun21':
		print('Not a valid observation date. Please enter again')
	else:
		pass
'''
obsdate = 0
while obsdate != 'May23':
	obsdate = str(input('What is the observation date for HD121617?\n(abbreviate month and do not add spaces): '))
	filestart = './HD121617'+obsdate+'_'
	if obsdate != 'May23':
		print('Not a valid observation date. Please enter again')
	else:
		pass


def FileCheck():
    if Path(filestart+'model_DMR.fits').is_file():
        return True
    elif Path(filestart+'model_map.fits').is_file():
        return True
    elif Path(filestart+'resid_DMR.fits').is_file():
        return True
    elif Path(filestart+'resid_map.fits').is_file():
        return True
    else:
        return False

if obsdate == 'May23':
    if FileCheck() == True:
        os.system('rm HD121617May23_*')
        print('\nAll files beginning with "HD121617May23_" have just been deleted')
    else:
        print('\nAll files beginning with "HD121617May23_" have already been deleted')
    if Path('ModelDiskMay23.fits').is_file():
        os.system('rm ModelDiskMay23.fits')
        print('\nYour model file has just been deleted')
    else:
        print('\nYour model file has already been deleted')
    print('\n<<<<<   Ready to run DMR   >>>>>\n')
else:
    print('\nNot a valid observation date for the disk HD121617')
'''
elif obsdate == 'Jun21':
    if FileCheck() == True:
        os.system('rm HD32297Jun21_*')
        print('\nAll files begging with "HD32297Jun21_" have just been deleted')
    else:
        print('\nAll files beginning with "HD32297Jun21_" have already been deleted')
    if Path('./ModelDiskJun21.fits').is_file():
        os.system('rm ModelDiskJun21.fits')
        print('\nYour model file has just been deleted')
    else:
        print('\nYour model file has already been deleted')
    print('\n<<<<<   Ready to run DMR   >>>>>\n')
else:
    print('\nNot a valid observation date for the disk HD121617')

'''
