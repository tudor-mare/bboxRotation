import math
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import copy


# three functions from Github that will help me rotate the images and the points
# the noise will be used to fill the black holes around the rotated image

def rotateImage(center, image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
  return result

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    cos = math.cos(angle)
    sin = math.sin(angle)

    x = (px - ox)
    y = (py - oy)

    qx = ox + cos * x - sin * y
    qy = oy + sin * x + cos * y
    return round(qx), round(qy)

def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

import random


# here we set how many angles we want our pictures to be rotated with
# 12 -> 360/12=30 -> 0,30,60, ..., 330
pictureNumber = 12
angles = np.linspace(0,360,pictureNumber,endpoint=False)

# set input/output directories
outputDir = "train_final11/"
inputDir = "train/"

# make an array of how much brighter or darker we want our pictures to be
brightnesses = [0.5,1,1.3]

j = 0

# we get the number of images in the folder so we make a progression percentage
# (half of the files are .xmls so we divide by 2)
len = len(os.listdir(inputDir))/2

for file in os.listdir(inputDir):
    if file.endswith(".png") or file.endswith(".jpg"):
        # we get the image name and extension

        fName , ending = os.path.splitext(file)
        # show percentage
        print ("{0:.1%}".format(j/len))
        j+=1

        # here we open the .xml file corresponding to the image
        root = ET.parse(os.path.join(inputDir,fName+'.xml')).getroot()
        tree = ET.ElementTree(root)
        bnd = root.find('object/bndbox')

        # here we get the coordinates of the top left point and
        # bottom right corner
        x1 = int(bnd.getchildren()[0].text)
        y1 = int(bnd.getchildren()[1].text)
        x3 = int(bnd.getchildren()[2].text)
        y3 = int(bnd.getchildren()[3].text)

        # we calculate top right corner point's coordonates - (x4,y4)
        # and bottom left point's coordonates - (x2,y2)
        x2 = x1
        y2 = y3

        x4 = x3
        y4 = y1

        # read the original picture
        orig = cv2.imread(os.path.join(inputDir,file),1);

        for angle in angles:
            for brightness in brightnesses:
                # we keep a deepcopy of the original image because we always
                # start rotating from the position of the original image
                img = copy.deepcopy(orig)

                # make a new layer with the same shape of our image
                # layer which we will put under our rotated picture to
                # fill the black holes the rotation leaves
                noua = np.zeros(img.shape)

                # we fill it with gaussian noise
                noua = noisy("gauss",noua)

                # our noise is valued between (-1,1) so we add 1
                # then we get (0,2) then multiply by 255 so we get
                # the range (0,510) and then divide by 2 so we get (0,255)
                noua = np.uint8((noua + 1) * 255 / 2)

                # make aux image as our image * brightness
                aux = brightness * img[:,:,:]

                # in case value go over 255 we cap them at 255
                aux[aux[:,:,:]>255] = 255

                # then convert our aux to uint8
                img[:,:,:] =np.uint8(aux)

                # calculate the middle of the bounding box
                centerX = (x1+x3)/2
                centerY = (y1+y3)/2


                # here we find where the new bounding box corners will end after rotating

                (x4N,y4N) = rotate((centerX,centerY),(x4,y4),math.radians(-angle))
                (x2N,y2N) = rotate((centerX,centerY),(x2,y2),math.radians(-angle))

                (x1N,y1N) = rotate((centerX,centerY),(x1,y1),math.radians(-angle))
                (x3N,y3N) = rotate((centerX,centerY),(x3,y3),math.radians(-angle))

                # we rotate the image
                img = rotateImage((centerX,centerY),img,angle)

                # we find the new topleft corner and bottom right corner
                xminNew = min(x1N,x3N,x4N,x2N)
                yminNew = min(y1N,y3N,y4N,y2N)

                xmaxNew = max(x1N,x3N,x4N,x2N)
                ymaxNew = max(y1N,y3N,y4N,y2N)

                # write in the .xml file
                bnd.getchildren()[0].text=str(xminNew)
                bnd.getchildren()[1].text=str(yminNew)
                bnd.getchildren()[2].text=str(xmaxNew)
                bnd.getchildren()[3].text=str(ymaxNew)


                # we make a mask of all the black pixels in our rotated image
                # so that we fill the black with noise
                black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)

                # here we fill
                img[black_pixels_mask] = noua[black_pixels_mask]

                # we convert the image from rgb to gray and back so we get
                # a grayscale image (which I discovered is better for my training)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

                # we get the root node and modify the name of the image in the .xml
                name = root.find('filename')
                name.text = fName+'-angle='+str(angle)+'-brightness='+str(brightness)+ending

                # here we write the new .xml and image with the same name so the other scripts recognize them easily
                # first we check if the directory exists, if not then we create one
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)
                tree.write(open(os.path.join(outputDir,fName+'-angle='+str(angle)+'-brightness='+str(brightness))+'.xml', 'wb'))
                cv2.imwrite(    os.path.join(outputDir,fName+'-angle='+str(angle)+'-brightness='+str(brightness))+ending, img)



print('Done!')
