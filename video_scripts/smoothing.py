import os
import numpy as np
import cv2 
from pathlib import Path
path_from = str(Path().absolute()) + "/smoothing_input/"

def average_frames(frames):
    # pad each side by an empty frame
    pad = np.zeros(frames[0].shape)
    pad_frames = np.array([pad] + frames + [pad])
    new_frames = np.uint8(pad_frames[:-2]/3 + pad_frames[1:-1]/3 + pad_frames[2:]/3)
    #f_new = cv2.cvtColor(f_new, cv2.COLOR_BGR2RGB)
    return new_frames


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
            new_frames = average_frames(frames)
            write_video(new_frames, name)