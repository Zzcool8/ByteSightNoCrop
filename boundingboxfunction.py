# This  function will take in an image, uses edging and class-based histogram thresholding to eliminate background and
# essentially zoom in on foreground object. In our case, this means cropping out any part of the image that is not a mosquito
# this is written with the aim to be further updated, with additional support for mosquito on non-blank paper.
# by Eduardo Sandoval and Laura Scavo, and Jewell for the structural framework, last edit 11242018
# with great help from Adam Goodwin, and the internet
# also used this to help with using the file name to rename the cropped file name
# https://stackoverflow.com/questions/44663347/python-opencv-reading-the-image-file-name

# to use boundingbox or boundingboxes


# for your own purposes I would change test file in main method
import cv2
import numpy as np
from thresholdingfunction import otsuthresholding
from objectidentificationnew2 import objectidentificationimage
import os
from classify import load_model, load_image_fromnumpy, predict_single
from torchvision import datasets, models, transforms
import torch
# honestly, still pretty new to classes, but I think this allows for the string conversion
# of this class type via the '__str__' method
class MyImage:
    def __init__(self, file):
        self.image = cv2.imread(file)
        self.__name = file

    def __str__(self):
        return self.__name

def boundingboxcoordinates(file):
    image = objectidentificationimage(file)
    pathofimage = str(MyImage(file))
    lastslash = pathofimage.rfind('/')
    filenameofimage = pathofimage[lastslash+1:]
    boundingboxcoordinates = {}
    boundingboxcoordinates[filenameofimage] = image[1]
    return boundingboxcoordinates


def boundingbox(file,modelfile,labelsfile):
    cropstatus =False
    uncropped = False
    filenameofimage = str(MyImage(file))
    lastslash = filenameofimage.rfind('/')
    output_filename = filenameofimage[:lastslash+1] + 'Cropped-' + filenameofimage[lastslash+1:]
    singleobjectimageparameters = objectidentificationimage(file)
    # thresholding, performed by otsuthresholding function
    # edgeimage = edges(file)
    # kernel = np.ones((6, 6), np.uint8)  # create 6x6 kernel, the option forces the elements to be an unsigned integer from 0 to 255
    # erosion and dilation provided for by cv2 function
    # erodedimage = cv2.erode(edgeimage, kernel, iterations=10)
    # dilatedimage = cv2.dilate(erodedimage, kernel, iterations=10)
    # next is to find our object boundaries
    # dimensions = np.shape(singleobjectimage)
    # y = dimensions[0]
    # x = dimensions[1]
    #initializing ...
    # max_height = 0
    # min_height = y
    # max_width = 0
    # min_width = x
    # check for the min and max of each coordinate axes, then we should have a bare minimum mosquito picture
    # as long we got rid of artifacts and noise
    # for i in range(y):
    #     for j in range(x):
    #         if singleobjectimage[i, j] == 0:
    #             if j < min_width:
    #                 min_width = j
    #             if j > max_width:
    #                 max_width = j
    #             if i < min_height:
    #                 min_height = i
    #             if i > max_height:
    #                 max_height = i
    #load in color image
    colorimage = cv2.imread(file)
    dimensions = np.shape(colorimage)
    croppingcoordinates = singleobjectimageparameters[1]
    skew = singleobjectimageparameters[2]
    mindistance = singleobjectimageparameters[3]
    print('1')
    croppedimage = colorimage[croppingcoordinates['min_height']:croppingcoordinates['max_height'], croppingcoordinates['min_width']:croppingcoordinates['max_width']]
    legroomwidth = int(abs(croppingcoordinates['min_width'] - croppingcoordinates['max_width']) * 0.05)
    legroomheight = int(abs(croppingcoordinates['max_height'] - croppingcoordinates['min_height']) * 0.05)
    #load in model and predict quality of cropped image using model
    model = load_model(modelfile, labelsfile)
    tensor_img = load_image_fromnumpy(croppedimage)
    resultlist = predict_single(model, tensor_img)
    index_max = np.argmax(resultlist)
    print('2')
    print(index_max)
    if index_max == 0:
        cropstatus = 'Bad'
    elif index_max == 2:
        while index_max == 2:
            croppingcoordinates['min_height'] = croppingcoordinates['min_height'] - legroomheight
            croppingcoordinates['max_height'] = croppingcoordinates['max_height'] + legroomheight
            croppingcoordinates['min_width'] = croppingcoordinates['min_width'] - legroomwidth
            croppingcoordinates['max_width'] = croppingcoordinates['max_width'] + legroomwidth
            print(croppingcoordinates)
            print(index_max)
            print('index_max')
            if croppingcoordinates['min_height'] < 0 or croppingcoordinates['max_height'] > dimensions[0] or croppingcoordinates['max_width'] > dimensions[1] or croppingcoordinates['min_width'] < 0:
                if filenameofimage[-5] == 'm':
                    croppedimage = colorimage
                    index_max = 1
                    cropstatus = 'Good'
                    uncropped = True
                else:
                    index_max = 1
                    cropstatus = 'Good'
            else:
                croppedimage = colorimage[croppingcoordinates['min_height']:croppingcoordinates['max_height'], croppingcoordinates['min_width']:croppingcoordinates['max_width']]
                tensor_img=load_image_fromnumpy(croppedimage)
                resultlist = predict_single(model, tensor_img)
                index_max = np.argmax(resultlist)
                print('index_max2')
                print(index_max)
                if index_max != 2:
                    cropstatus = 'Good'
    else:
        cropstatus = 'Good'
    #ratiodistance = min(float(int(mindistance)/int(dimensions[0])), float(int(mindistance)/int(dimensions[1])))
    #if skew < 0.45 or ratiodistance > 0.5:
    #    cropstatus = 'Bad'
    #else:
    #    cropstatus = 'Good'
    # might wanna change this as well, if other images types being used
    #
    # next line writes image, can comment out or not
    #
    #
    #cv2.imwrite(output_filename, croppedimage,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # image = cv2.resize(croppedimage, (1200, 600))
    # cv2.imshow('croppedimage',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return croppedimage, croppingcoordinates, cropstatus, uncropped


#https://stackoverflow.com/questions/22207936/python-how-to-find-files-and-skip-directories-in-os-listdir
# used this with an assist in parsing through files in directories
def boundingboxes(folder,modelfile,labelsfile):
    #imagesandcoordinates = {}
    print(folder)
    lastslash = folder.rfind('/') #find last slash to create folders and text files out of the folder of images
    newpath = folder[0:lastslash]
    print(newpath)
    txtfile = open(os.path.join(newpath,'boundingboxes.txt'), 'w')
    txtfile.write('file    min_height     max_height    min_width   max_width\n')
    os.mkdir(newpath + '/Bad')
    os.mkdir(newpath + '/Good')
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            continue
        else:
            print(file)
            txtfile.write(file + '    ')
            # special character '/' is dependent on windows directories
            try:
                uncropped = False
                coordinatesofimage = boundingbox(folder + '/' + file, modelfile, labelsfile)
                croppedimage = coordinatesofimage[0]
                coordinates = coordinatesofimage[1]
                cropstatus = coordinatesofimage[2]
                txtfile.write(str(coordinates['min_height'])+'    ')
                txtfile.write(str(coordinates['max_height'])+'    ')
                txtfile.write(str(coordinates['min_width'])+'    ')
                txtfile.write(str(coordinates['max_width'])+'    ')
                txtfile.write(cropstatus)
                txtfile.write('\n')
                filenameofimage = str(MyImage(file))
                lastslash = filenameofimage.rfind('/')
                if uncropped == False:
                    output_filename = newpath + '/' + cropstatus + '/Cropped-' + filenameofimage[lastslash+1:]
                else:
                    output_filename = newpath + '/' + cropstatus + filenameofimage[lastslash + 1:]
                print(output_filename)
                cv2.imwrite(output_filename, croppedimage,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except KeyError:
                txtfile.write("didn't take \n")
         #imagesandcoordinates.update(coordinatesofimage)
    txtfile.close()
    #return imagesandcoordinates


def main():
    test_file = "C:/Users/edsan/Documents/Mosquito Lab Code/Lab Code/newpictures/JHU-004430_01m.jpg"
    test_file2= 'C:/Users/edsan/Documents/Mosquito Lab Code/Lab Code/Test Folder/Test Folder 2/JHU-005334_02p.jpg'
    #boundingbox(test_file)
    #boundingbox(test_file2)
    #folderpath = "E:/VectorWeb/R&D/testforfcropping"
    #folderpath2 = "C:/Users/edsan/Documents/Mosquito Lab Code/Lab Code/Test Folder/TestFOLDER3"
    folderpath5 ="/home/vectorweb4/Documents/Development_Sandbox/Eduardo/emailself/AedesAegyptifailures/JHU-000385_05d.jpg"
    #folderpath3 = "C:/Users/edsan/Documents/Mosquito Lab Code/Lab Code/newpictures"
    folderpath4 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAlbopictus3/AedesAlbopictus3"
    folderpath5 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAegypti3/AedesAegypti3"
    folderpath6 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/testwithmodel/testwithmodel"
    folderpath7 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/CulexQuinquefasciatus3"
    modelfile = '/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/Cropknowledge/_res/model.pb'
    labelsfile = '/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/Cropknowledge/_res/labels.txt'
    #boundingboxes(folderpath7,modelfile,labelsfile)
    boundingboxes(folderpath5, modelfile, labelsfile)
    boundingboxes(folderpath4, modelfile, labelsfile)
    # print(boundingboxescoordinates)
if __name__ == '__main__':
    main()
