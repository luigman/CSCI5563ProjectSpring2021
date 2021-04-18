from PIL import Image
import numpy as np
import os
from pathlib import Path
import cv2

path_from = str(Path().absolute().parent) + "/shading_albedo/"
path_reconstruct = str(Path().absolute()) + "/reconstructed_images/"
path_color =  str(Path().absolute()) + "/recolored_images/"

# Create uncorrected reconstruction
def reconstruct(albedo_dict, shading_dict):
    reconstructedimages = []
    keys = []
    for key in albedo_dict.keys():
        img1 = Image.open(path_from + albedo_dict[key])
        img2 = Image.open(path_from + shading_dict[key])
        image1 = np.array(img1)
        image2 = np.array(img2)
        image1 = image1 / 255
        new_img = np.uint8(np.rint(image1*image2) + 1)
        new_image = Image.fromarray(new_img)
        new_image.save(path_reconstruct + key + '-reconstruct.jpg')
        reconstructedimages.append(new_img)
        keys.append(key)
    return reconstructedimages, keys

# recolor images
def recolor(img):
    green = (255/2.4)/np.mean(img[:, :, 1])
    blue = (255/2.4)/np.mean(img[:, :, 2])
    red = (255/2.4)/np.mean(img[:, :, 0])
    new_img = np.copy(img)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if new_img[i][j][1] * green < 255:
                new_img[i][j][1] *= green
            else:
                new_img[i][j][1] = 255
            if new_img[i][j][2] * blue < 255:
                new_img[i][j][2] *= blue
            else:
                new_img[i][j][2] = 255
            if new_img[i][j][0] * red < 255:
                new_img[i][j][0] *= red
            else:
                new_img[i][j][0] = 255
    new_img = np.uint8(new_img)
    #new_img = hisEqulColor(new_img)
    return new_img

def hisEqulColor(img):
    #Histogram Equalization from https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

if __name__ == '__main__':
    # Read in files from shading and albedo folder
    shading_dict = {}
    albedo_dict = {}
    for file in os.listdir(path_from):
        filename = file.split("-")
        if len(filename) == 2:
            if filename[1][0] == 'a':
                albedo_dict[filename[0]] = file
            elif filename[1][0] == 'b':
                shading_dict[filename[0]] = file
            else:
                print('Error: improperly named file ' + str(file))
    # reconstruct images
    reconstructedimages, keys = reconstruct(albedo_dict, shading_dict)
    # recolor images
    for i in range(len(keys)):
        new_img = recolor(reconstructedimages[i])
        new_img = Image.fromarray(new_img)
        new_img.save(path_color + keys[i] + 'recoloredimg.jpg')
    