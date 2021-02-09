# This code will take in a picture and threshold it
# It will obtain the black and white ratio
# It will do this for all pictures
#
from thresholdingfunction import otsuthresholding
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#line 62 and line 13, currently testing dilations
def blackandwhiteratio(thresholded_image_numpyarray):
    dimensions = np.shape(thresholded_image_numpyarray)
    print(dimensions)
    unique, counts = np.unique(thresholded_image_numpyarray, return_counts=True)
    print(unique)
    if len(unique) == 2:
        for i in range(len(unique)):
            if unique[i] == 0:
                blacklevel = counts[i]
                print(counts[i])
            elif unique[i] == 255:
                whitelevel = counts[i]
                print(counts[i])
            else:
                print('Nonmaximally white or black pixel found')
        if blacklevel + whitelevel == dimensions[0]*dimensions[1]:
            whiteblackratio = float(whitelevel)/float(blacklevel)
            print(whiteblackratio)
        else:
            print('wrong')
            whiteblackratio = False
    elif len(unique) == 1:
        print('There is only one pixel found')
        whiteblackratio = False
    else:
        print('There are issues with this code')
        whiteblackratio = False
    return whiteblackratio

def bwratios(folder):
    lastslash = folder.rfind('/')
    folder2 = folder[0:lastslash]
    secondtolastslash = folder2.rfind('/')
    dataset = folder2[secondtolastslash+1:] + ' ' + folder[lastslash+1:]
    listofbwratios = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            continue
        else:
            print(file)
            full_filename = folder + '/' + file
            print(full_filename)
            try:
                bwratio = blackandwhiteratio(full_filename)
                listofbwratios.append(bwratio)
            except IndexError:
                continue
            except UnboundLocalError:
                continue
    listofbwratios = np.array(listofbwratios)
    print(listofbwratios)
    plt.hist(listofbwratios, bins=np.arange(min(listofbwratios), max(listofbwratios) + 5, 5))
    mean = np.mean(listofbwratios)
    std = np.std(listofbwratios)
    plt.title("White/Black ratios of dilated" + str(dataset))
    plt.xlabel('black and white ratios')
    plt.ylabel('Counts')
    plt.savefig(str(dataset) + ": " + str(mean) + ", " + str(std) + ".png")
    plt.clf()
    print(mean,std)
    return mean, std
def main():
    testfile1 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/tlboxes/Cropped-JHU-000022_01p.jpg"
    testfolder1 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/tlboxes"
    testfolder2 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAlbopictus3/tlboxes"
    testfolder3 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAegypti3/tlboxes"
    testfolder4 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/zoomout"
    testfolder5 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/perfect"
    testfolder6 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAlbopictus3/zoomout"
    testfolder7 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAlbopictus3/perfect"
    testfolder8 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAegypti3/zoomout"
    testfolder9 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAegypti3/perfect"
    #blackandwhiteratio(testfile1)
    bwratios(testfolder1)
    bwratios(testfolder2)
    bwratios(testfolder3)
    bwratios(testfolder4)
    bwratios(testfolder5)
    bwratios(testfolder6)
    bwratios(testfolder7)
    bwratios(testfolder8)
    bwratios(testfolder9)

if __name__ == '__main__':
    main()