# Generate Diffuse Map from Environment Map
# https://learnopengl.com/PBR/IBL/Diffuse-irradiance
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Convert vector/sphere coord to 2d coords
def hemisphere_to_img_crd(sampleVec):
    #Checks to see if sample_vec is outside of hemisphere range
    if sampleVec[2] < 0:
        return [np.nan , np.nan]
    sampleVec = sampleVec * 128.0
    sampleVec = sampleVec + 128.0
    rounded = np.around(sampleVec)
    if rounded[0] > 255 or rounded[0] < 0:
        #round down
        rounded[0] = 255
    if rounded[1] > 255 or rounded[1] < 0:
        #round down
        rounded[1] = 255
    light_map_crds = np.asarray([rounded[0], rounded[1]]).astype(int)
    return light_map_crds

def img_crd_to_hemisphere(img_crd):
    #shift img_crd to zero at center
    coord = img_crd - 128
    #divide by radius to get between -1 and 1
    coord = coord / 128.0
    #sphere crd = (x,y,sqrt((1-x^2+y^2))
    sphere_coord = (coord[0], coord[1], np.sqrt(1-(np.square(coord[0]) + np.square(coord[1]))))
    return sphere_coord

#get one point of diffuse map from light map
#adapted from: https://learnopengl.com/PBR/IBL/Diffuse-irradiance
def diffuse_from_light_for_point(direction, light_map, sampleDelta):
    point_diffuse = np.zeros(3)
    up = [0.0,0.0,1.0]
    right = (np.cross(up, direction))
    up = (np.cross(direction, right))
    direction = np.asarray(direction)
    nr_samples = 0
    phi = 0
    while phi < math.pi * 2.0:
        theta = 0
        while theta < math.pi / 2.0:
            tangentSample = np.asarray([math.sin(theta) * math.cos(phi),  math.sin(theta) * math.sin(phi), math.cos(theta)])
            sampleVec = tangentSample[0] * right + tangentSample[1] * up + tangentSample[2] * direction
            index = hemisphere_to_img_crd(sampleVec)
            if not np.isnan(index[0]):
                point_diffuse += light_map[index[0]][index[1]] * math.cos(theta) * math.sin(theta)
                nr_samples += 1
            theta += sampleDelta
        phi += sampleDelta
    point_diffuse = math.pi * point_diffuse * (1/nr_samples)
    return point_diffuse

#get diffuse map from environment map
def diffuse_from_light(light_map, sampleDelta):
    diffuse_map = np.zeros(np.shape(light_map))
    sphere_coords_x = []
    sphere_coords_y = []
    sphere_coords_z = []
    for i in range(0,len(light_map)):
        print('{} / 255'.format(i))
        for j in range(0,len(light_map[0])):
            #Filter out non circle parts
            if not np.isnan(light_map[i][j][0]):
                #From pixel coords get circle coords/direction
                sphere_coord = img_crd_to_hemisphere(np.asarray([i,j]))
                sphere_coords_x.append(sphere_coord[0])
                sphere_coords_y.append(sphere_coord[1])
                sphere_coords_z.append(sphere_coord[2])
                #get individual diffuse points
                point_diffuse = diffuse_from_light_for_point(sphere_coord, light_map, sampleDelta)
                diffuse_map[i][j] = point_diffuse
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(sphere_coords_x, sphere_coords_y, sphere_coords_z, s=1)
    plt.savefig("sphere.png")

    return diffuse_map


# EDIT VARIABLES HERE
image_name = 'dir_23_chrome256.jpg'
sample_rate = 0.4
blur_size = 10

#read image
img = cv2.imread(image_name)
img = img.astype(float)

# Sets non circle values to nan
for i in range(0,len(img)):
    for j in range(0,len(img[0])):
        temp_i = i - 128
        temp_j = j - 128
        if np.sqrt(np.square(temp_i) + np.square(temp_j)) > 128:
            img[i][j] = [np.nan, np.nan, np.nan]

# MAIN CALL
im_diffuse = diffuse_from_light(img, sample_rate)

# Set nans to 0
im_diffuse = np.nan_to_num(im_diffuse)

# find average pixel value
total_sum = [0,0,0]
num = 0
for i in range(0,len(img)):
    for j in range(0,len(img[0])):
        temp_i = i - 128
        temp_j = j - 128
        if np.sqrt(np.square(temp_i) + np.square(temp_j)) < 128:
            total_sum += im_diffuse[i][j]
            num += 1

average_val = total_sum / num
average_val = np.around(average_val)

#set 0 values to average to cleare empty values
im_diffuse[np.all(im_diffuse == (0, 0, 0), axis=-1)] = average_val

#blur image
im_diffuse = cv2.blur(im_diffuse,(blur_size,blur_size))

# Sets non circle values to [0,0,0]
for i in range(0,len(img)):
    for j in range(0,len(img[0])):
        temp_i = i - 128
        temp_j = j - 128
        if np.sqrt(np.square(temp_i) + np.square(temp_j)) > 128:
            im_diffuse[i][j] = [0, 0, 0]


cv2.imwrite('diffuse.png', im_diffuse)
