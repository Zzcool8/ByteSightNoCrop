# Eduardo Sandoval
# Last update: April 3rd, 2019 
#
# Since, the script will be long-running, and have a lot of data, this script assumes results.txt file has already been created
# Tidbits that may be important:
# Should there be a results file for each species?
# Should results themselves be ordered, will this make it easier to search for duplicates?
# How to avoid duplicates
# For now, since the 'good' category is not 100% reliable, and so is the 'fixed' category.
# I will perform the following:
# 1. I will check the good and fixed categories after running scripts and obtaining statistics
# 2. I will remove bad eggs.
# 3. My code will check only images taken from the good files, and output their bounding box by matching with the text file
# 4. As my 'good' label becomes more robust, we eliminate step 3 and simply check the bounding box text file for all
# 'good' pictures
# thoughts running through head
# also needs to be able to delete images from overarching dataset of mosquitos
# Current: outputtingresults, takes boundingbox.txt and folder with images and checks
# that folder contains image, and then adds coordinates to resultspath from boundingbox, assuming
# that coordinates are listed in boundingbox.txt. Note this can be used to obtain coordinates for badly cropped images as well

# first import all necessary modules and packages
import os
import re
import cv2

def outputtingresults(folder,boundingboxpath,resultspath):
    # first, I open file with bounding box information as a read only
    boundingboxfilehandle = open(boundingboxpath,'r')
    boundingboxfile = boundingboxfilehandle.read()
    #bbmm = mmap.mmap(boundingboxfile.fileno(), 0, mmap.PROT_READ)
    # I check if results.txt file exists, and create one if it doesn't
    if os.path.isfile(resultspath) == True:
        resultsfile = open(resultspath,'a+')
    else:
        resultsfile = open(resultspath,'a+')
        resultsfile.write('file    min_height     max_height    min_width   max_width cropstatus\n')
    # I comb through files in folder of good pictures
    directoryname = os.path.dirname(resultspath)
    # will only work on ubuntu but I'm lazy
    character = directoryname.rfind('/')
    resultsstring = resultsfile.read()
    for file in os.listdir(folder):
        filename = file[-18:]
        print(filename)
        path = os.path.join(folder, file)
        if os.path.isdir(path):
            continue
        else:
            # for each file, I write the filename in results.txtfile
            # search through bounding box text file
            # I also check that the file and its coordinates is not already in the results.txt
            # ad hoc solution that might fail when the file gets too big but for now is whatever
            for line in boundingboxfile.splitlines():
                if filename in line:
                    presence = re.search(filename,resultsstring)
                    print(presence)
                    if presence == None:
                        resultsfile.write(line + '\n')
                        #resultsfile.write('    ' + directoryname[character+1:] + '\n')
                        resultsfile.flush()
                        resultsstring = resultsstring + line + '\n'
                        break
                    else:
                        print('File is already accounted for')
                        break
    boundingboxfilehandle.close()
    resultsfile.close()


# It's called remove miscope but it simply removes images of a certain type and lumps them into a new folder
# Note: this only works because we place an 'm' 'd' and 'p' before the extension of each image file


def removemiscope(folder,charactertype):
    # purpose: remove all miscope files and place them in a separate folder.
    print(folder)
    # first generate a new folder in same directory that will contain all miscopes
    directoryname = os.path.dirname(folder)
    character = len(directoryname)
    actualfolder = folder[character:]
    newdirectorypath = directoryname + actualfolder + charactertype
    try:
        os.mkdir(newdirectorypath)
    except FileExistsError:
            pass
    for file in os.listdir(folder):
        fileextension = file.rfind('.')
        # our naming convention is put type of photo in the character before the file extension
        if file[fileextension-1] == charactertype:
            print(file)
            # Note that in both cases the directory in which the new file is being created must already exist, (but, on Windows, a file with that name must not exist or an exception will be raised).
            # I.e. this is less robust in windows
            os.rename(folder + '/' + file, newdirectorypath + '/' + file)
        else:
            continue

#file    min_height     max_height    min_width   max_width


# photocheck takes boundingbox.txt and big data set folder and uses coordinates to write a new image
# sanity check for after running cropping



def photocheck(folder,textfile):
    # This file will read a bounding box test file and attempt to find each picture that has coordinates attached. After finding
    # and loading said picture, it will crop the picture using said coordinates and load it.
    # This will most likely be used to test for correct bounding box info, and very likely only on my computer because I don't know
    # how to do this besides using cv2.imshow(). Wait nevermind, I can use cv2.imwrite
    directoryname = os.path.dirname(folder)
    newdir = directoryname + '/photocheck'
    try:
        os.mkdir(newdir)
    except FileExistsError:
        pass
    txtfile = open(textfile,'r')
    actualfile = txtfile.read()
    for line in actualfile.splitlines():
        if 'Good' or 'Fixed' in line:
            character = line.find(' ')
            print(line[0:character])
            try:
                image = cv2.imread(folder + '/' + line[0:character])
                print(image[0,0])
            except TypeError:
                print('Error:' + line[0:character])
                print('1234')
                continue
            # now that we've found the picture, we should attempt to retrieve the coordinates
            coordinates = [1,2,3,4]
            newline = line[character+4:]
            for i in range(4):
                newcharacter = newline.find(' ')
                try:
                    coordinates[i] = int(newline[0:newcharacter])
                    newline = newline[newcharacter+4:]
                except ValueError:
                    break
            croppedimage = image[coordinates[0]:coordinates[1],coordinates[2]:coordinates[3]]
            outputfilename = newdir + '/' + line[0:character]
            cv2.imwrite(outputfilename,croppedimage,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    txtfile.close()


# After results file has been obtained, removefromdataset takes in the large dataset folder and
# removes each image that is present in the resultsfile
# This ensures no images will be uneccessarily cropped



def removefromdataset(folder,resultsfile):
    # Function will read images from final results file(output of outputting results)
    #directoryname = os.path.dirname(folder)
    txtfile = open(resultsfile, 'r')
    actualfile = txtfile.read()
    for line in actualfile.splitlines():
        character = line.find(' ')
        ogimage = line[0:character]
        try:
            os.remove(folder + '/' + ogimage)
        except FileNotFoundError:
            continue




# If anyone ever needs to use this, outputting results takes in set of good photos 'sampleset', and a textfile of boundingboxes 'boundingboxpath'
# and generates a resultsfile that must be prespecified 'resultsfile' and puts only the bounding boxes of the good photos in the results
# photocheck plots cropped photos using bounding box
# removefromdataset, removes the files with bounding box in resultsfile so that pictures that were cropped well, are not used again
# finally removemiscope can be used to separate out files,
def main():
    boundingboxpath = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/boundingboxes.txt"
    resultspath = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/results.txt"
    goodpicturesfolder = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/Good"
    #outputtingresults(goodpicturesfolder, boundingboxpath, resultspath)
    sampleset = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesAelbopictus/Good"
    dataset = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesAelbopictus/AedesAlbopictusTotalp"
    dataset2 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AnophelesGambiaeTotal/AnophelesGambiaeTotal/arabiensis"
    boundingboxpath2 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesAelbopictus/MedianBlurtlboundingboxes.txt"
    resultsfile = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images_current/AedesAelbopictus/resultsculexerraticus.txt"
    #outputtingresults(sampleset,boundingboxpath2,resultsfile)
    #photocheck(dataset,resultsfile)
    #removefromdataset(dataset,resultsfile)
    removemiscope(dataset2,'m')
    #removemiscope(dataset,'d')

if __name__ == '__main__':
    main()