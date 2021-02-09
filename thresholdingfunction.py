# by Eduardo Sandoval
# June 21st, 2018
# This program will begin with converting a grey scale image file into a single column vector,
# then plots the histogram of the values contained, i.e. the frequencies of each respective
# value, then establishes several different methods of dynamic global thresholding, on an
# image by image basis,
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
#gaussian filter
import numpy as np
import cv2
# takes in numpy array instead of image file now
def otsuthresholding(img):
    dimensions = np.shape(img)
    print(dimensions)
    linearized = np.reshape(img, dimensions[0]*dimensions[1])
    bin = range(max(linearized)+1)
    hist, bins = np.histogram(linearized, bins = bin)
    #plt.hist(linearized, bins=bin)
    #plt.show()
    minvariance = 100000000000000
    bestthreshold = 2
    for i in bins:
        weightone = (sum(hist[0:i]))/len(linearized)+1
        weighttwo = (sum(hist[i:len(bins)-1]))/len(linearized)+1
        dummylist = np.array(range(0,i))
        histogramarray = np.array(hist[0:i])
        classonemean = sum(dummylist*histogramarray)/(len(hist)*weightone)
        classtwomean = sum(hist[1:len(bins)])/(len(hist)*weighttwo)
        variance = weightone*weighttwo*((classonemean - classtwomean)**2)
        if variance < minvariance:
            minvariance = variance
            bestthreshold = i
        else:
            continue
    for i in range(len(linearized)):
        if linearized[i] >= bestthreshold:
            linearized[i] = 255
        else:
            linearized[i] = 0
    # print(linearized)
    thresholdimg = np.reshape(linearized,dimensions)
    # image = cv2.resize(thresholdimg, (1200, 600))
    # ims = cv2.imshow('image',image)
    # cv2.waitKey(0)
    print('minimum variance:' + str(minvariance))
    print('best threshold:' + str(bestthreshold))
    return thresholdimg





def main():
    test_file3 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/emailself/AAeF/JHU-000384_05d.jpg"
    test_file2 =  "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAegypti3/AedesAegypti3/JHU-000384_03p.jpg"
    test_file = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/CulexQuinquefasciatus3/CulexQuinquefasciatus3/JHU-000641_04p.jpg"
    test_file4 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/problemimages/JHU-000596_03d.jpg"
    test_file5 = "/home/vectorweb4/Documents/Development_Sandbox/Eduardo/for_HCL/test_images/AedesAlbopictus3/AedesAlbopictus3/JHU-000084_07p.jpg"
    image = cv2.imread(test_file5,0)
    thresholdimage = otsuthresholding(image)
    lastslash = test_file5.rfind('/')
    #output_filename = test_file5[0:lastslash] + '/Thresholded' + test_file5[lastslash+1:]
    #cv2.imwrite(output_filename, thresholdimage, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    image = cv2.resize(thresholdimage, (1600, 800))
    cv2.imshow('thresholdimage',image)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
# figure out how to eliminate foreground from offcenter
