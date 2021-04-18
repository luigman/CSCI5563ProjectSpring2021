import cv2
import numpy as np
import scipy
from scipy import ndimage
import os
import sys
sys.path.append('../reconstruction')
from reconstruction import recolor
import matplotlib.pyplot as plt

def uvMapping(n):
    #Maps a normal vector to the sphere map
    #Sphere mapping equation: https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node177.html
    n[2]=-n[2] #opengl assumes camera looks in -z direction
    r = n
    p = np.sqrt(r[0]**2+r[1]**2+(r[2])**2) #website says there should be +1 here, but I don't think we need it for our coordinate system

    s = r[0]/p
    t = r[1]/p
    mag = np.sqrt(s**2 + t**2)
    valid = mag <= 0.97
    if not valid:
        s = s*0.97
        t = t*0.97
    s = s/2+1/2
    t = t/2+1/2
    
    s_px = (s)*255
    t_px = (t)*255

    return t_px, s_px #swap coordinates for row-priority indexing

def uvMapping1(n,u):
    #Maps the reflected vector to the sphere map
    #Sphere mapping equation: https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node177.html
    n[2]=-n[2] #opengl assumes camera looks in -z direction
    r = u - 2*(n.dot(u))*n
    p = np.sqrt(r[0]**2+r[1]**2+(r[2]+1)**2)

    s = r[0]/p
    t = r[1]/p
    mag = np.sqrt(s**2 + t**2)
    valid = mag <= 1
    if not valid:
        print("Invalid Point")
        s = s/mag
        t = t/mag
    s = s/2+1/2
    t = t/2+1/2
    
    s_px = (s)*255
    t_px = (t)*255

    return s_px, t_px

def relight(img1, lgt2):
    stride = 2
    #lgt2 = ndimage.uniform_filter(lgt2+np.ones(lgt2.shape))#+0.001
    img2_out = np.zeros((img1.shape[0], img1.shape[1], 3))
    texture_coverage = np.zeros((256,256))
    for i in range(0,img1.shape[0], stride):
        for j in range(0,img1.shape[1], stride):
            n = nrm1[i,j]
            n = n/np.linalg.norm(n)
            u = np.linalg.inv(K_apprx)@np.array([i,j,1]) #Ray from camera center to u
            u = u/np.linalg.norm(u)
            
            s_px, t_px = uvMapping(n)

            img2_out[i:i+stride,j:j+stride] = lgt2[int(s_px),int(t_px)]
            texture_coverage[int(s_px),int(t_px)] = 255
            
    return img2_out, texture_coverage

def convertSpec(spec):
    #A rough approximation of what Isaac is doing
    #Just takes the envoronment map, removes black pixels and blurs it
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            u = i/127.5-1 #convert to [-1,1]
            v = j/127.5-1
            r = np.sqrt(u**2+v**2)
            theta = np.arctan2(u,v)
            if r>1:
                u = (np.sin(theta)+1)*127.5
                v = (np.cos(theta)+1)*127.5
                spec[i,j] = spec[int(u),int(v)]
    spec=ndimage.uniform_filter(spec,size=(50,50,1))
    # plt.imshow(spec)
    # plt.show()
    return spec


if __name__ == '__main__':
    inputs = ['00004_00034_indoors_150_000', '00039_00294_outdoor_240_000', 'everet_dining1', 'main_d424-12', 'willow_basement_21']
    lights = ['dir_0','dir_2','dir_3','dir_5','dir_19','dir_22','dir_24']

    for im in inputs:
        print("Processing image",im)
        img1 = cv2.imread('input/'+im+'/albedo.jpg')
        shading_gt = cv2.imread('input/'+im+'/shading.jpg')
        shading_gt = cv2.cvtColor(shading_gt, cv2.COLOR_BGR2GRAY)
        
        shading_3 = np.zeros(img1.shape)
        shading_3[:,:,0] = shading_gt
        shading_3[:,:,1] = shading_gt
        shading_3[:,:,2] = shading_gt
        shading_gt = shading_3

        nrm1 = cv2.imread('input/'+im+'/normal.png')
        nrm1 = cv2.inpaint(nrm1,np.uint8((nrm1==0).all(axis=2)),3,cv2.INPAINT_TELEA)
        #nrm1 = ndimage.uniform_filter(nrm1,size=3)
        nrm1 = cv2.resize(cv2.cvtColor(nrm1, cv2.COLOR_BGR2RGB), (img1.shape[1], img1.shape[0])) #not sure why this shape is backwards

        #Convert normals to coordinate system from class
        #(z away from camera, x to the right, y down)
        nrm1 = nrm1[:,:,(1,0,2)]
        nrm1 = (nrm1/127.5-1)
        nrm1[:,:,:2] = -nrm1[:,:,:2]

        #new network:
        # nrm1 = nrm1[:,:,(2,0,1)]
        # nrm1 = (nrm1/127.5-1)
        # nrm1[:,:,:2] = -nrm1[:,:,:2]
        # nrm1 = (nrm1+1)*127.5
        # plt.imshow(np.uint8(nrm1))
        # plt.show()

        # Approximate K
        f = 300
        fov_y = 36
        fov_x = 52
        fy = -img1.shape[0]/(2*np.tan(fov_y/2))
        fx = img1.shape[1]/(2*np.tan(fov_x/2))
        K_apprx = np.asarray([
            [fx, 0, img1.shape[0]/2], #change to fx for non-cropped images
            [0, fy, img1.shape[1]/2],
            [0, 0, 1]
        ])

        if not os.path.exists('output/'+im):
            os.makedirs('output/'+im+'/')

        cv2.imwrite('output/'+im+'/shading_gt.png', shading_gt)

        relit_gt = shading_gt/255*img1
        relit_gt = recolor(relit_gt)
        cv2.imwrite('output/'+im+'/img_reconst.png', relit_gt)

        for light in lights:
            print("    Processing light",light)
            spec = cv2.imread('input/lights/'+light+'_chrome256.jpg')
            diff = cv2.imread('input/lights/'+light+'_gray256.jpg')
            spec = convertSpec(np.array(spec))
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            shading, texture_coverage = relight(img1, diff)
            cv2.imwrite('output/'+im+'/shading'+light+'_out.png', shading)
            cv2.imwrite('output/'+im+'/textture_coverage.png', texture_coverage)

            specular, texture_coverage = relight(img1, spec)
            cv2.imwrite('output/'+im+'/spec'+light+'_out.png', specular)

            relit = shading/255*img1
            relit = recolor(relit)
            cv2.imwrite('output/'+im+'/relit_diff'+light+'.png', relit)
            relit = specular/255*img1
            relit = recolor(relit)
            cv2.imwrite('output/'+im+'/relit_spec'+light+'.png', relit)