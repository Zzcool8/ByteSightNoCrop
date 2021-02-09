# Eduardo Sandoval
# Last Update: May 20th
# This code will ideally re-sort all mosquito foldered data by location, instead of by species



# 1. Need to read in excel file with all relevant data.
import pandas as pd
import os

filename = "C:/Users/edsan/Downloads/JHI Image Database_2018_08_22.xlsx"
inlinefile = pd.read_excel(open(filename,'rb'),sheet_name='Mosquito Table')
# 2. Specify a folder, and for each picture find corresponding official ID in excel file
# 3. After finding official ID in excel file, obtain three or four digit code
# 4. After obtaining code, either find folder made, or generate folder
folder = "D:/AedesAegyptiSample1"
folderlist = set()
nestedfolderindex = folder.rfind('/')
#newfolder = folder[0:nestedfolderindex]
newfolder = os.path.dirname(folder)
for root,dirs,files in os.walk(folder):
    for file in files:
        print(file[0:-8])
        for row in inlinefile.itertuples():
            matchfile = row[1]
            if file[0:-8] == matchfile:
                mosquitoid = row[2]
                areacodeindex = mosquitoid.find('-')
                areacode = mosquitoid[0:areacodeindex]
                print(areacode)
                if areacode in folderlist:
                    newfolderpath = os.path.join(newfolder,areacode)
                    currentpath = os.path.join(root,file)
                    newpath = os.path.join(newfolderpath,file)
                    os.rename(currentpath, newpath)
                    continue
                else:
                    folderlist.add(areacode)
                    newfolderpath = os.path.join(newfolder,areacode)
                    os.mkdir(newfolderpath)
                    currentpath = os.path.join(root,file)
                    newpath = os.path.join(newfolderpath,file)
                    os.rename(currentpath, newpath)
                break
            else:
                continue

print(folderlist)
