from PIL import Image
from skimage import exposure
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
def recolor_hist(img, orig_img):
    multi = True if img.shape[-1] > 1 else False
    new_img = exposure.match_histograms(img, orig_img, multichannel=multi)
    return new_img

def recolor_normalize(img, orig_img):
    green_mean = np.mean(orig_img[:, :, 1])
    blue_mean = np.mean(orig_img[:, :, 2])
    red_mean = np.mean(orig_img[:, :, 0])
    green_std = np.std(orig_img[:, :, 1])
    blue_std = np.std(orig_img[:, :, 2])
    red_std = np.std(orig_img[:, :, 0])
    green_mean_new = np.mean(img[:, :, 1])
    blue_mean_new = np.mean(img[:, :, 2])
    red_mean_new = np.mean(img[:, :, 0])
    green_std_new = np.std(img[:, :, 1])
    blue_std_new = np.std(img[:, :, 2])
    red_std_new = np.std(img[:, :, 0])
    new_img = np.copy(img)

    new_img[:,:,0] = (img[:,:,0]-red_mean_new)/red_std_new
    new_img[:,:,1] = (img[:,:,1]-green_mean_new)/green_std_new
    new_img[:,:,2] = (img[:,:,2]-blue_mean_new)/blue_std_new
    new_img[:,:,0] = new_img[:,:,0]*red_std+red_mean
    new_img[:,:,1] = new_img[:,:,1]*green_std+green_mean
    new_img[:,:,2] = new_img[:,:,2]*blue_std+blue_mean
    new_img[new_img>255]=255
    new_img[new_img<0]=0

    new_img = np.uint8(new_img)
    #new_img = hisEqulColor(new_img)
    return new_img

def recolor(img, orig_img):
    green_mean = np.mean(orig_img[:, :, 1])
    blue_mean = np.mean(orig_img[:, :, 2])
    red_mean = np.mean(orig_img[:, :, 0])
    green = green_mean/np.mean(img[:, :, 1])
    blue = blue_mean/np.mean(img[:, :, 2])
    red = red_mean/np.mean(img[:, :, 0])
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
    