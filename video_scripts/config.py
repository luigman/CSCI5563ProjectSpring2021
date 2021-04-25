import argparse
class BaseOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.parser = parser
        parser.add_argument("--video_name",default="face_video.mp4")
        parser.add_argument("--image",default=None)
        parser.add_argument("--lighting",default=None)
        parser.add_argument("--intrinseg_weights_loc",default="../intrinsic_image_decomposition/intrinseg/experiment/synthetic_trained/checkpoints/final.checkpoint")
        parser.add_argument("--normal_weights_loc",default="./normal_weights/ExpNYUD_three.ckpt")
        parser.add_argument("--cmap_file_loc",default="./normal_weights/cmap_nyud.npy")
        parser.add_argument("--output_file",default="relit_video.mp4")
        parser.add_argument("--visualize",nargs='?', default=False, const=True)
        parser.add_argument("--benchmark",nargs='?', default=False, const=True)
        parser.add_argument("--gt_normals",nargs='?', default=False, const=True)
        parser.add_argument("--gt_lighting",nargs='?', default=False, const=True)
        parser.add_argument("--kinect",nargs='?', default=False, const=True)