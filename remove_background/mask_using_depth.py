import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# ------------------- masked rcnn setup ---------------------------

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "remove_background/coco/"))  # To find local version
import coco

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)


# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# ------------------- Function to Remove Background ---------------------------
def mask_background(mask_model, color_name, depth_name, threshold):

    #image = skimage.io.imread('dir_7_mip2.jpg')
    image = skimage.io.imread(color_name)

    # crop image to square (as depth image is square)
    length = np.floor(np.shape(image)[0] / 2)
    center_col = np.floor(np.shape(image)[1] / 2.0)
    center_range = [int(center_col - length), int(center_col + length)]
    image = image[:,center_range[0]:center_range[1]]

    # Run detection
    results = mask_model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    # Uncomment to visualize all masks
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    # plt.savefig('all_masks.jpg')
    # plt.close()

    #laad depth
    image_depth = skimage.io.imread(depth_name)

    # match depth imsize to color size
    image_depth = skimage.transform.resize(image_depth, (np.shape(image)[0], np.shape(image)[1]))

    # average depth values of masks and if above threshold combine and show them
    average = np.zeros(len(r['class_ids']))
    masks = np.transpose(r['masks'], (2,0,1))
    close_list = []
    for i in range(0,len(r['class_ids'])):
        masked_im = np.multiply(image_depth, masks[i])
        average[i] = masked_im[np.nonzero(masked_im)].mean()
        print("Average Depth of Each Detected Object:")
        print(average[i])
        if average[i] < threshold:
            close_list.append(i)

    masks_close = masks[close_list]

    complete_mask = np.logical_or.reduce(masks_close)

    masked_im_final = np.multiply(image_depth, complete_mask)
    # Uncomment to visualize final masks
    #skimage.io.imsave('final_masks.png',masked_im_final)

    mask3 = np.transpose([masked_im_final, masked_im_final, masked_im_final], (1,2,0))
    final_image = np.multiply(image, mask3)

    return final_image


# -------------------- MAIN CALL -----------------------------
masked_im = mask_background(model, 'dir_7_mip2.jpg', 'test2_depth_consistency.png', 0.07)
skimage.io.imsave('masked_im.png',masked_im)
