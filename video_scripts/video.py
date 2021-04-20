import sys
sys.path.append('../reconstruction')
sys.path.append('../relighting')
sys.path.append('../intrinsic_image_decomposition/intrinseg')
import cv2
import numpy as np
import copy
import av
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from PIL import Image
from reconstruction import *
from relighting import *
from direct_intrinsics_sn import DirectIntrinsicsSN
from infer import main, set_experiment
from utils import Cuda, create_image
from normal_weights.models import net as normal_net
from config import BaseOptions

print("Import Successful")
print("PyTorch can see",torch.cuda.device_count(),"GPU(s). Current device:",torch.cuda.current_device())

inst =  BaseOptions()
parser = inst.parser
opt = parser.parse_args()

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

def get_decomposition(img):

    net_in = ['rgb']
    net_out = ['albedo','shading','segmentation']
    normalize = True
    net = DirectIntrinsicsSN(3,['color','color','class'])
    net.load_state_dict(torch.load(opt.intrinseg_weights_loc))
    cuda = Cuda(0)
    net = net.cuda(device=cuda.device)
    net.eval()

    copy_im = copy.deepcopy(img)
    pil_im = Image.fromarray(copy_im)
    resized_im = pil_im.resize((480,352),Image.ANTIALIAS)  # IntrinSeg network only accepts input of this size
    resized_im = np.array(resized_im,dtype=np.int64)
    resized_im = resized_im.astype(np.float32)
    resized_im[np.isnan(resized_im)] = 0
    in_ = resized_im
    in_ = in_.transpose((2, 0, 1))

    if normalize:
        in_ = (in_ * 255 / np.max(in_)).astype('uint8')
        in_ = (in_ / 255.0).astype(np.float32)

    in_ = np.expand_dims(in_, axis=0)
    rgb = torch.from_numpy(in_)
    rgb = Variable(rgb).cuda(device=cuda.device)
    albedo_out, shading_out, segmentation_out = net(rgb)
    albedo = albedo_out.cpu().detach().numpy()
    shading = shading_out.cpu().detach().numpy()
    albedo_out = create_image(albedo)
    shading_out = create_image(shading)

    return albedo_out,shading_out

def get_normals(img):
    CMAP = np.load(opt.cmap_file_loc)
    DEPTH_COEFF = 5000. # to convert into metres
    HAS_CUDA = torch.cuda.is_available()
    MAX_DEPTH = 8.
    MIN_DEPTH = 0.
    NUM_CLASSES = 40
    NUM_TASKS = 3 # segm + depth + normals
    model = normal_net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
    if HAS_CUDA:
        _ = model.cuda()
    _ = model.eval()
    ckpt = torch.load(opt.normal_weights_loc)
    model.load_state_dict(ckpt['state_dict'])
    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if HAS_CUDA:
        img_var = img_var.cuda()
    segm, depth, norm = model(img_var)
    norm = cv2.resize(norm[0].cpu().data.numpy().transpose(1, 2, 0),img.shape[:2][::-1],interpolation=cv2.INTER_CUBIC)
    out_norm = norm / np.linalg.norm(norm, axis=2, keepdims=True)
    return out_norm


if __name__ == "__main__":

    """
    Open the video file and start frame-by-frame processing
    """

    output = av.open(opt.output_file,'w')
    stream = output.add_stream('mpeg4',24)
    stream.bit_rate = 8000000


    video = av.open(opt.video_name)

    k = 0

    for frame in video.decode(video=0):
        img = frame.to_image()
        img = np.asarray(img)

        """
        Calculate albedo, shading and normals
        """

        albedo,shading_gt = get_decomposition(img)
        albedo = np.asarray(albedo)
        shading_gt = np.asarray(shading_gt)
        shading_gt = cv2.cvtColor(shading_gt,cv2.COLOR_BGR2GRAY)

        normals = get_normals(img)
        lights = ['dir_0']

        """
        Relight Image
        """

        lights = ['dir_23']
        shading_3 = np.zeros_like(albedo)
        for i in range(3):
            shading_3[:,:,i] = shading_gt
        
        nrm1 = convertNormalsNew(normals)

        # Approximate K
        f = 300
        fov_y = 36
        fov_x = 52
        fy = -albedo.shape[0]/(2*np.tan(fov_y/2))
        fx = albedo.shape[1]/(2*np.tan(fov_x/2))
        K_apprx = np.asarray([
            [fx, 0, albedo.shape[0]/2], #change to fx for non-cropped images
            [0, fy, albedo.shape[1]/2],
            [0, 0, 1]
        ])

        for light in lights:
            print("Using light: %s" % (light) )
            spec = cv2.imread("input/lights/%s_chrome256.jpg" % (light))
            diff = cv2.imread("input/lights/%s_gray256.jpg" % (light))
            spec = convertSpec(np.array(spec))
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

            shading, diff_coverage = relight(albedo,diff, nrm1, K_apprx)
            shading = recolor_normalize(shading, shading_3)
            #specular, spec_coverage = relight(albedo,spec, nrm1, K_apprx)

            relit_diff = (shading/255)*albedo
            relit_diff = recolor_normalize(relit_diff,img)

            #relit_spec = (specular/255)*albedo
            #relit_spec = recolor(relit_spec,albedo)
        print(k)
        k += 1
        frame = av.VideoFrame.from_ndarray(relit_diff, format='rgb24')
        packet = stream.encode(frame)
        output.mux(packet)
    
    output.close()
    


        

