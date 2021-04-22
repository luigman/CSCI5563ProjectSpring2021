import os
import numpy as np
import cv2 
from pathlib import Path
import matplotlib.pyplot as plt
path_from = str(Path().absolute()) + "/smoothing_input/"

def average_frames_vid(frames):
    # pad each side by an empty frame
    pad = np.zeros(frames[0].shape)
    pad_frames = np.array([pad] + frames + [pad])
    new_frames = np.uint8(pad_frames[:-2]/3 + pad_frames[1:-1]/3 + pad_frames[2:]/3)
    #f_new = cv2.cvtColor(f_new, cv2.COLOR_BGR2RGB)
    return new_frames

def average_frames(frames):
    h, w, _ = frames[-1].shape
    new_frame = np.zeros(frames[0].shape)
    for i, frame in enumerate(frames):
        new_frame = new_frame + frame
    
    new_frame = new_frame / (i+1)
    return new_frame

def average_frames_warp(frames):
    #Image warping from: https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    h, w, _ = frames[-1].shape
    new_frame = np.zeros(frames[0].shape)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)
    
    numFrames = np.zeros(frames[-1][:,:,0].shape)
    for i, frame in enumerate(frames):
        (cc, warp_matrix) = cv2.findTransformECC(frame[:,:,0],frames[-1][:,:,0],np.eye(3, 3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, criteria,np.ones(frame[:,:,0].shape).astype(np.uint8),5)
        frame = cv2.warpPerspective(frame, warp_matrix, (w,h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # plt.imshow(frame)
        # plt.show()
        new_frame = new_frame + frame
        numFrames[(frame!=0).all(axis=2)] +=1
    numFrames_3 = np.zeros(new_frame.shape)
    for c in range(3):
        numFrames_3[:,:,c] = numFrames
    numFrames_3[numFrames_3==0] = 1
    new_frame = new_frame / numFrames_3
    return new_frame

def video_frames(video):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    frames = []
    while(1):
        frames.append(frame)
        ret, frame = cap.read()
        if ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
    return frames

def write_video(new_frames, vid_name):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    size = new_frames[0].shape[0:2]
    writer = cv2.VideoWriter(name, fourcc, 24, (size[1], size[0]))
    for frame in new_frames:
        writer.write(frame)

if __name__ == '__main__':
    for file in os.listdir(path_from):
        filename = file.split(".")
        if len(filename) == 2:
            name = 'smoothing_output/' + filename[0] + '_new.avi'
            frames = video_frames('smoothing_input/' + file)
            new_frames = average_frames_vid(frames)
            write_video(new_frames, name)