# Object Identification and iterative transfer learning guided check of found objects
# by Eduardo Sandoval, last update June 19, 2019
# Line 400 add stuff to rerun code
# Line 313 fix stuff by adding code, specifically want to make it so bands are added sequentially, and not 4 bands at once
import cv2
import numpy as np
from thresholdingfunction import otsuthresholding
from classify import load_model, load_image_fromnumpy, predict_single
from torchvision import datasets, models, transforms
import torch
import os
from blackandwhiteratios import blackandwhiteratio

#from PIL import Image


class MyImage:
    def __init__(self, file):
        self.image = cv2.imread(file)
        self.__name = file

    def __str__(self):
        return self.__name
###############################################################################
# This function takes in an image file, model file, corresponding labels file, the steps,
# and scale, and outputs a list of coordinates for a cropped mosquito, the cropped
# mosquito image in an array format, and the percieved status(Good, Bad, Truncated, False, and Fixed), and
# image slices of all 'objects' perceived as mosquitos.
# In slightly more detail, this function reads in an image file in an array format,
# thresholds the picture as well as other preprocessing steps to allow for a cleaned up image
# comprising of two values, 0 and 1. We then loop through the array and check for black values and their neighbors
# to define objects within the pictures. After preliminary discrimination of the objects,
# such as removing those objects with a small size, we load in a transfer learning
# network that has been trained previously to identify photos of perfectly
# cropped pictures, truncated pictures where part of the mosquito is found
# and badly cropped pictures. All objects seen as mosquitos are saved, to hopefully use as samples in a future
# training of the transfer learning network. The object most likely to be a mosquito(highest prediction score)
# is taken as the actual mosquito. Some space is added in case, mosquito legs were removed during pre-processing.
#
###############################################################################
# 'scale' is a hyperparameter, as is steps
def tlobjectidentification(image_file,modelfile,labelsfile, steps,scale):
    ###############################################################################
    # First we obtain the filename of the image. I know there are cleaner ways to do this
    # but I didn't know that at the time
    # os.listdir(file) would probably be more useful and more general than this
    # ad-hoc method
    ###############################################################################
    pathofimage = str(MyImage(image_file))
    lastslash = pathofimage.rfind('/')
    filenameofimage = pathofimage[lastslash + 1:]
    objectdictionary = {}
    ###############################################################################
    # Next, we read in our image and we threshold the image, followed by
    # image pre-processing. Image pre-processing consists of a median blur, and one single
    # erosion and dilation using a 5*5 kernel of ones.
    # Finally, if the image is a miscope image (I'm aware this is hard coded, sue me)
    # then we define a different windowSize and looping
    ###############################################################################
    objectimage = cv2.imread(image_file,0)
    blurredimage = cv2.medianBlur(objectimage,5)
    objectimage2 = otsuthresholding(blurredimage)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(objectimage2, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    objectimage = img_dilation
    dimensions = np.shape(objectimage)
    if filenameofimage[-4] == 'm':
        steps = 1
        skip = True
        windowSize = int(2*scale * min(dimensions))
        looping = [dimensions[0],dimensions[1]]
    else:
        skip = False
        windowSize = int(scale * min(dimensions))
        looping = [int(dimensions[0]/steps), int(dimensions[1]/steps)]
    number = 0
    pixelvalues = np.unique(objectimage)
    print(pixelvalues)
    maximumpixelvalue = max(pixelvalues)
    bwimage = np.zeros(dimensions,np.uint8) + objectimage
    ################################################################################
    # Now we have defined our windowSize and looping. Our steps hyperparameter is an
    # integer n. We use this to only look at a pixel every n steps, i.e. if it was 30,
    # we would look at pixel 1, then pixel 31. This speeds up the computation greatly,
    # and doesn't affect the accuracy that much, at least qualitatively.
    # Here it gets confusing, at least to me. So imagine we get to our first 0 value pixel.
    # We define a 'searchzone' around this pixel, the size of which is determined by the
    # windowSize hyperparameter. We obtain a list of all pixel values within this searchzone,
    # and since this is our first 0 value pixel, we would expect to obtain a list containing
    # only 255 and 0, again because we binarized the image previously. Next we obtain all coordinates
    # within the search-zone that have a pixel value of 0, including our OG center pixel by looping
    # through the search-zone(there is definitely a better way to do this, like maybe
    # [i for i, x in enumerate(searchzoneproper)]
    # but honestly, this might not work and I'm not sure whether this will be faster at all.
    # Anywho, now we have an image with mostly 255 and 0 values, and some 1 values. We continue to loop
    # through the image and find another 0 value pixel. We generate our searchzone, obtain a list
    # of all unique values, and remove our trivial values 0 and 255. There are three main cases:
    # 1. Our list contains no other values. This suggests that an entirely new object has been discovered
    # In our case, we would then define all 0 values in this searchzone as '2' or whatever object we're up to.
    # Side note: We are limited to 255 objects, but who wants to deal with that large/cluttered
    # of an image that needs more than 255 objects defined?
    # 2. Our list contains one other value. Up to this point, the only possibility is '1'.
    # The pixel we are centered on, and all surrounding 0 pixels are then considered to be part of object '1'.
    # 3. Our list contains more than one other value. Not possible up till now, but imagine further in the loop
    # when we have let's say 5 objects defined, and maybe '1' and '4' pixel values are found. Then perhaps, 1 and 4
    # are not actual separate objects, but one large object. We compress these objects into one single object(in a messy
    # way).
    # Side note: This encourages zoomed-out crops, since a disembodied leg or
    # artifact that is close enough to the mosquito will cause it to be recognized
    # as a part of the object. To fix this, it might be worth it to rerun the
    # function on the resulting cropped image. The caveat here will be that the
    # bounding box coordinates are no longer correct, and thus will need to be
    # corrected by the coordinates obtained by the first run.
    ###############################################################################
    for j in range(looping[0]):
        for i in range(looping[1]):
            coordinates = [j*steps,i*steps]
            # changed != maximumpixelvalue to == 0, to see if this is bit faster
            # faulty logic is the reason,
            if objectimage[coordinates[0], coordinates[1]] == 0:
                y = coordinates[0] - windowSize
                x = coordinates[1] - windowSize
                if y < 0 and x < 0:
                    x = 0
                    y = 0
                elif y < 0:
                    y = 0
                elif x < 0:
                    x = 0
                # Define an area of search zone
                searchzoneproper = objectimage[y:j*steps + windowSize, x:i*steps + windowSize]
                objectsnearpixel = list(np.unique(searchzoneproper))
                try:
                    objectsnearpixel.remove(255)
                except ValueError:
                    pass
                try:
                    objectsnearpixel.remove(0)
                except ValueError:
                    pass
                if len(objectsnearpixel) == 0:
                    number += 1
                    #objects.append(number)
                    objectimage[coordinates[0], coordinates[1]] = number
                    objectdictionary[number] = [(coordinates[0], coordinates[1])]
                    newdimensions = np.shape(searchzoneproper)
                    for l in range(newdimensions[0]):
                        for m in range(newdimensions[1]):
                            if searchzoneproper[l, m] == 0:
                                calculatedcoordinates = (y+l,x+m)
                                objectimage[calculatedcoordinates[0],calculatedcoordinates[1]] = number
                                objectdictionary[number].append(calculatedcoordinates)
                    continue
                elif len(objectsnearpixel) == 1:
                    newnumber = objectsnearpixel[0]
                    objectimage[coordinates[0],coordinates[1]] = newnumber
                    objectdictionary[newnumber].append((coordinates[0],coordinates[1]))
                    continue
                elif len(objectsnearpixel) > 1:
                    print('at least two objects found')
                    newnumber = min(objectsnearpixel)
                    print(newnumber)
                    objectimage[coordinates[0],coordinates[1]] = newnumber
                    objectdictionary[newnumber].append((coordinates[0],coordinates[1]))
                    print(objectsnearpixel)
                    for k in objectsnearpixel:
                        print(k)
                        if k == newnumber:
                            continue
                        else:
                            secondaryobject = objectdictionary[k]
                            print(secondaryobject)
                            #secondaryobject = list(secondaryobject)
                            for z in range(len(secondaryobject)):
                                #print(z)
                                #print(secondaryobject[z])
                                objectdictionary[newnumber].append(secondaryobject[z])
                                changedcoordinates = secondaryobject[z]
                                objectimage[changedcoordinates[0],changedcoordinates[1]] = newnumber
                            # fix this in future
                            del objectdictionary[k]
                            continue
            else:
                continue
    print(str(len(objectdictionary)) + ' objects in image: Checking for mosquitos now')
    ###############################################################################
    # Hooray, we now have a dictionary of all objects found in image, and the pixels
    # that comprise each object. We load in our transfer learning model to have it
    # predict the nature of each object detected. We determine a thresholdarea(magic
    # numbers again) to filter our objects, and remove those that are most likely
    # noise. Finally, we generate a list of coordinates to crop our image based on
    # the pixels comprising the object. We use the network to predict whether
    # the cropped image contains a mosquito. There is also a block of commented
    # code, in case you want to see the objects highlighted on the original image
    # in real time. It might not work, but it did at some point, in a different
    # environment.
    model = load_model(modelfile, labelsfile)
    colorimage = cv2.imread(image_file)
    coordinates = {}
    coordinates['miny'] = []
    coordinates['maxy'] = []
    coordinates['minx'] = []
    coordinates['maxx'] = []
    scores = []
    if filenameofimage[-4] == 'm':
        thresholdarea = 50
    elif filenameofimage[-4] == 'p':
        thresholdarea = 100
    else:
        thresholdarea = 500
    for k in objectdictionary:
        print(k)
        #objectcoordinates = objectdictionary[k]
        #        help on sorting tuples found here
        # https://stackoverflow.com/questions/14802128/tuple-pairs-finding-minimum-using-python
        objectcoordinates = np.array(objectdictionary[k])
        if len(objectcoordinates) < thresholdarea:
            continue
        minx = dimensions[1]
        maxx = 0
        maxy = 0
        miny = dimensions[0]
        for (yy, xx) in objectdictionary[k]:
            if xx <= minx:
                minx = xx
            if xx >= maxx:
                maxx = xx
            if yy <= miny:
                miny = yy
            if yy >= maxy:
                maxy = yy
        if minx == maxx:
            continue
        if maxy == miny:
            continue
        print(k)
        croppedimage = colorimage[miny:maxy+1,minx:maxx+1]
        #cv2.imshow('image',croppedimage)
        #cv2.waitKey(500)
        #newerimage2 = cv2.resize(newimage, (800, 400))
        #cv2.imshow('image', newerimage2)
        #cv2.waitKey(500)
        #clone = image.copy()
        #clone = cv2.rectangle(clone, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        #clone2 = cv2.resize(clone, (800,400))
        #cv2.imshow("Window", clone2)
        #cv2.waitKey(700)
        tensor_img = load_image_fromnumpy(croppedimage)
        resultlist = predict_single(model, tensor_img)
        index_max = np.argmax(resultlist)
        print(index_max)
        ###############################################################################
        # With the most likely class being produced, we then have three cases.
        # 1. Index == 0, The first class is a bad crop. We just skip and move to the
        # next object.
        # 2. Index == 1, The second class is a good crop. This is the best case
        # scenario. Here, we simply append the bounding box coordinates to our master
        # list of coordinates.
        # 3. Index == 2, The third and final class is a truncated crop. This suggests
        # that
        ###############################################################################
        if index_max == 0:
            continue
        elif index_max == 1:
            scores.append(resultlist[1])
            print('found one')
            coordinates['miny'].append(miny)
            coordinates['maxy'].append(maxy)
            coordinates['minx'].append(minx)
            coordinates['maxx'].append(maxx)
        elif index_max == 2:
            scores.append(resultlist[2])
            print('found truncated')
            bands = 10
            miny = miny - bands
            maxy = maxy + bands
            minx = minx - bands
            maxx = maxx + bands
            coordinates['miny'].append(miny)
            coordinates['maxy'].append(maxy)
            coordinates['minx'].append(minx)
            coordinates['maxx'].append(maxx)
    if len(coordinates['miny']) == 0:
        print('this method failed')
        newcoordinates = False
        status = 'False'
    elif len(coordinates['miny']) == 1:
        print('mosquito found')
        newcoordinates = {}
        newcoordinates['miny'] = min(coordinates['miny'])
        newcoordinates['maxy'] = max(coordinates['maxy'])
        newcoordinates['minx'] = min(coordinates['minx'])
        newcoordinates['maxx'] = max(coordinates['maxx'])
        status = 'Good'
    else:
        print('too many candidates')
        bestobject = np.argmax(scores)
        print(bestobject)
        # this is new
        # only use when needing new data to train with
        #
        #
        #for w in range(len(coordinates['miny'])):
        #    coordinatecycleminx = coordinates['minx'][w]
        #    coordinatecyclemaxx = coordinates['maxx'][w]
        #    coordinatecycleminy = coordinates['miny'][w]
        #    coordinatecyclemaxy = coordinates['maxy'][w]
        #    images.append(colorimage[coordinatecycleminx:coordinatecyclemaxx+1,coordinatecycleminy:coordinatecyclemaxy+1])
        #
        newcoordinates = {}
        newcoordinates['miny'] = coordinates['miny'][bestobject]
        newcoordinates['maxy'] = coordinates['maxy'][bestobject]
        newcoordinates['minx'] = coordinates['minx'][bestobject]
        newcoordinates['maxx'] = coordinates['maxx'][bestobject]
        status = 'Fixed'
    if newcoordinates == False:
        croppedimage = colorimage
    else:
        width = newcoordinates['maxx']-newcoordinates['minx']
        height = newcoordinates['maxy']-newcoordinates['miny']
        someroom = int(max(height,width)*.1)
        newcoordinates['miny'] -= someroom
        newcoordinates['maxy'] += someroom
        newcoordinates['minx'] -= someroom
        newcoordinates['maxx'] += someroom
        for key , value in newcoordinates.items():
            if value <= 0:
                newcoordinates[key] = 0
            else:
                continue
        croppedimage = colorimage[newcoordinates['miny']:newcoordinates['maxy'],newcoordinates['minx']:newcoordinates['maxx']]
    #print(newcoordinates)
    #print(croppedimage)
    #print(status)
    return newcoordinates, croppedimage, status


def tlboundingboxes(folder,modelfile,labelsfile,steps,scale):
    print(folder)
    lastslash = folder.rfind('/')  # find last slash to create folders and text files out of the folder of images
    newpath = folder[0:lastslash]
    print(newpath)
    bbtxtfile = 'MedianBlurtlboundingboxes.txt'
    txtfile = open(os.path.join(newpath, bbtxtfile), 'w')
    txtfile.write("file    min_height     max_height    min_width   max_width status \n")
    try:
        os.mkdir(newpath + '/ZoomedOut')
    except OSError:
        pass
    try:
        os.mkdir(newpath + '/Good')
    except OSError:
        pass
    try:
        os.mkdir(newpath + '/False')
    except OSError:
        pass
    try:
        os.mkdir(newpath +'/Fixed')
    except OSError:
        pass
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            continue
        else:
            print(file)
            txtfile.write(file + '    ')
            # special character '/' is dependent on windows directories
            try:
                parameters = tlobjectidentification(folder + '/' + file, modelfile, labelsfile,steps,scale)
                print(parameters)
                tlimage = parameters[1]
                status = parameters[2]
                filenameofimage = str(MyImage(folder + '/' + file))
                lastslash = filenameofimage.rfind('/')
                #output_filename = newpath + '/tlboxes' + '/Cropped-' + filenameofimage[lastslash + 1:]
                #print(output_filename)
                #cv2.imwrite(output_filename, tlimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                coordinates = parameters[0]
                if status == 'Good':
                    output_filename = newpath + '/Good' + '/Cropped-' + filenameofimage[lastslash + 1:]
                    cv2.imwrite(output_filename, tlimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                elif status == 'False':
                    output_filename = newpath + '/False' + '/Cropped-' + filenameofimage[lastslash+1:]
                    cv2.imwrite(output_filename,tlimage,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                elif status == 'Fixed':
                    output_filename = newpath +'/Fixed' + '/Cropped-' + filenameofimage[lastslash+1:]
                    cv2.imwrite(output_filename, tlimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                else:
                    output_filename = newpath + '/ZoomedOut' + '/Cropped-' + filenameofimage[lastslash + 1:]
                    cv2.imwrite(output_filename, tlimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(output_filename)
                txtfile.write(str(coordinates['miny']) + '    ')
                txtfile.write(str(coordinates['maxy']) + '    ')
                txtfile.write(str(coordinates['minx']) + '    ')
                txtfile.write(str(coordinates['maxx']) + '    ')
                txtfile.write(str(parameters[2]))
                txtfile.write('\n')
            except ValueError:
                txtfile.write("Value: didn't take \n")
            except KeyError:
                txtfile.write("Key: didn't take \n")
            except TypeError:
                txtfile.write("Type: didn't take \n")
    txtfile.close()

def main():
    test_file = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/emailself/AAlF/JHU-000215_02d.jpg"
    modelfile = '/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/Transfer_Learning_previous_networks/Cropknowledge_res_3/model.pb'
    labelsfile = '/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/Transfer_Learning_previous_networks/Cropknowledge_res_3/labels.txt'
    #test = tlobjectidentification(test_file, modelfile, labelsfile, 1)
    folderpath4 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesAelbopictus/AedesAlbopictusTotalp"
    folderpath5 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesDorsalisTotal/AedesDorsalisTotal"
    folderpath6 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/CulexPipiensTotal/CulexPipiensTotal"
    #folderpath7 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/PsorophoraCiliataTotal/PsorophoraCiliataTotal"
    folderpath8 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/CulexErraticusTotal/CulexErraticusTotal"
    folderpath9 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/PsorophoraColumbiaeTotal/PsorophoraColumbiaeTotal"
    #folderpath10 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesJaponicusTotal/AedesJaponicusTotal"
    folderpath11 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesSollicitansTotal/AedesSollicitansTotal"
    folderpath12 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesTaeniorhynchusTotal/AedesTaeniorhynchusTotal"
    folderpath13 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesVexansTotal/AedesVexansTotal"
    folderpath14 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/PsorophoraCyanescensTotal/PsorophoraCyanescensTotal"
    folderpath15 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AnophelesFunestusTotal/AnophelesFunestusTotal/ss"
    folderpath16 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AnophelesGambiaeTotal/AnophelesGambiaeTotal/arabiensis"
    #newdata(folderpath7, modelfile, labelsfile,1,35,35)
    #newdata(folderpath4, modelfile, labelsfile, 1, 35, 35)
    tlboundingboxes(folderpath4, modelfile, labelsfile,40, 0.08)
    #tlboundingboxes(folderpath5, modelfile, labelsfile,20, 0.04)
    #tlboundingboxes(folderpath8, modelfile, labelsfile,40, 0.1)
    tlboundingboxes(folderpath6, modelfile, labelsfile,20, 0.04)
    #tlboundingboxes(folderpath9, modelfile, labelsfile,20, 0.08)
    #tlboundingboxes(folderpath11,modelfile,labelsfile,40, 0.1)
    #tlboundingboxes(folderpath12, modelfile, labelsfile, 1, 0.07)
    #tlboundingboxes(folderpath13, modelfile, labelsfile, 40, 0.07)
    #tlboundingboxes(folderpath14, modelfile, labelsfile, 5, 0.05)
    tlboundingboxes(folderpath15, modelfile, labelsfile,20, 0.08)
    tlboundingboxes(folderpath16, modelfile, labelsfile,20, 0.08)
if __name__ == '__main__':
    main()
