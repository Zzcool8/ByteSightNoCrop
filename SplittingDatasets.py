import pandas as pd
import os
import random
from functools import reduce

filename = "C:/Users/edsan/Downloads/JHI Image Database_2018_08_22.xlsx"
inlinefile = pd.read_excel(open(filename,'rb'),sheet_name='Mosquito Table')
# 2. Specify a folder, and for each picture find corresponding official ID in excel file
# 3. After finding official ID in excel file, obtain three or four digit code
# 4. After obtaining code, either find folder made, or generate folder
folder = "D:\LocationData"
folderlist = set()
nestedfolderindex = folder.rfind('/')
#newfolder = folder[0:nestedfolderindex]
newfolder = os.path.dirname(folder)



filenames = {'train':set(),'val':set(),'test':set()}
trainfold = os.path.join(newfolder,'train')
testfold = os.path.join(newfolder,'test')
valfold = os.path.join(newfolder,'val')
os.mkdir(trainfold)
os.mkdir(testfold)
os.mkdir(valfold)
folds = [trainfold,testfold,valfold]
newfolder = os.path.dirname(folder)
for root,dirs,files in os.walk(folder):
    rootpath = os.path.dirname(root)
    for directory in dirs:
        print(directory)
        for i in range(len(folds)):
            newdirectory = os.path.join(folds[i],directory)
            os.mkdir(newdirectory)
            continue
    for file in files:
        print(file)
        filename = file[0:-8]
        print(filename)
        print(filenames.values)
        if filename in filenames['train']:
            currentpath = os.path.join(root,file)
            index = root.rfind('\\')
            print(root[index+1:])
            newpath = os.path.join(newfolder,'train',root[index+1:],file)
            os.rename(currentpath, newpath)
            continue
        elif filename in filenames['test']:
            currentpath = os.path.join(root,file)
            index = root.rfind('\\')
            print(root[index+1:])
            newpath = os.path.join(newfolder,'test',root[index+1:],file)
            os.rename(currentpath, newpath)
            continue
        elif filename in filenames['val']:
            currentpath = os.path.join(root,file)
            index = root.rfind('\\')
            print(root[index+1:])
            newpath = os.path.join(newfolder,'val',root[index+1:],file)
            os.rename(currentpath, newpath)
            continue
        else:
            number = random.random()
            if number < float(0.1):
                newset = filenames['test']
                newset.add(filename)
                filenames['test'] = newset
                currentpath = os.path.join(root,file)
                index = root.rfind('\\')
                print(root[index+1:])
                newpath = os.path.join(newfolder,'test',root[index+1:],file)
                os.rename(currentpath, newpath)
                continue
            elif  float(0.1) <= number <= float(0.3):
                newset = filenames['val']
                newset.add(filename)
                filenames['val'] = newset
                currentpath = os.path.join(root,file)
                index = root.rfind('\\')
                print(root[index+1:])
                newpath = os.path.join(newfolder,'val',root[index+1:],file)
                os.rename(currentpath, newpath)
                continue
            else:
                newset = filenames['train']
                newset.add(filename)
                filenames['train'] = newset
                currentpath = os.path.join(root,file)
                index = root.rfind('\\')
                print(root[index+1:])
                newpath = os.path.join(newfolder,'train',root[index+1:],file)
                os.rename(currentpath, newpath)
                continue
