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
from smoothing import average_frames, average_frames_warp
from scipy import ndimage

print("PyTorch can see",torch.cuda.device_count(),"GPU(s). Current device:",torch.cuda.current_device())

inst =  BaseOptions()
parser = inst.parser
opt = parser.parse_args()

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

def initNetworks():
    #Decomposition network
    net = DirectIntrinsicsSN(3,['color','color','class'])
    net.load_state_dict(torch.load(opt.intrinseg_weights_loc))
    cuda = Cuda(0)
    net = net.cuda(device=cuda.device)
    net.eval()

    #Normals network
    HAS_CUDA = torch.cuda.is_available()
    NUM_CLASSES = 40
    NUM_TASKS = 3 # segm + depth + normals
    model = normal_net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
    if HAS_CUDA:
        _ = model.cuda()
    _ = model.eval()
    ckpt = torch.load(opt.normal_weights_loc)
    model.load_state_dict(ckpt['state_dict'])

    return net, model

def prepare_img(img):
    img = img*IMG_SCALE
    IMG_MEAN = np.mean(img, axis=(0,1)).reshape((1, 1, 3))
    IMG_STD = np.std(img, axis=(0,1)).reshape((1, 1, 3))
    return (img - IMG_MEAN) / IMG_STD

def get_decomposition(img, net):
    net_in = ['rgb']
    net_out = ['albedo','shading','segmentation']
    normalize = True
    cuda = Cuda(0)

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

    albedo_out = cv2.resize(np.asarray(albedo_out),(img.shape[1],img.shape[0]))
    shading_out = cv2.resize(np.asarray(shading_out),(img.shape[1],img.shape[0]))
    shading_out = cv2.cvtColor(shading_out,cv2.COLOR_BGR2GRAY)

    return albedo_out,shading_out

def get_normals(img,out_shape,model):
    NUM_CLASSES = 40
    CMAP = np.load(opt.cmap_file_loc)
    HAS_CUDA = torch.cuda.is_available()

    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if HAS_CUDA:
        img_var = img_var.cuda()
    segm, depth, norm = model(img_var)
    segm = segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0)
    segm = CMAP[segm.argmax(axis=2) + 1].astype(np.uint8)

    norm = cv2.resize(norm[0].cpu().data.numpy().transpose(1, 2, 0),out_shape[:2][::-1],interpolation=cv2.INTER_CUBIC)
    out_norm = norm / np.linalg.norm(norm, axis=2, keepdims=True)
    return out_norm

def visualize(albedo,shading_gt,normals,shading_pre,shading,relit):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(shading_gt, cmap='gray')
    axs[0].title.set_text("Predicted Shading")
    axs[1].imshow(shading_pre)
    axs[1].title.set_text("Relit Shading")
    axs[2].imshow(shading)
    axs[2].title.set_text("Normalized Shading")
    plt.show()

    normals = visualizeNormals(normals)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(albedo, cmap='gray')
    axs[0].title.set_text("Predicted Albedo")
    axs[1].imshow(normals, cmap='gray')
    axs[1].title.set_text("Predicted Normals")
    axs[2].imshow(relit, cmap='gray')
    axs[2].title.set_text("Relit image")
    plt.show()

    return

def loadDataset():
    '''
    Load the multiillumination dataset
    Returns:
    frames_list: list of images to relight (shape numScenes*numProbes)
    lights_list: list of lighting conditions (shape numScenes*numProbes x numProbes)
    '''
    dataset_dir = './input/multi_dataset'
    img_names = [name for name in os.listdir(dataset_dir)]
    img_list = []
    lights_list = []
    for img in img_names:
        
        img_filenames = [os.path.join(dataset_dir,img,name) for name in os.listdir(os.path.join(dataset_dir,img)) if name.endswith('mip2.jpg')]
        
        img_list.append(img_filenames)
        light_filenames = []
        for img_file in img_filenames:
            light_num = img_file.split('_')[-2]
            light_filename = os.path.join(dataset_dir,img,'probes','dir_'+light_num)
            assert os.path.isfile(light_filename+'_gray256.jpg')
            light_filenames.append(light_filename)
            lights_list.append(light_filenames)
    
    return [item for sublist in img_list for item in sublist], lights_list

def loadKinect(lights):
    '''
    Load the multiillumination dataset

    Returns:
    frames_list: list of images to relight (shape numScenes*numProbes)
    lights_list: list of lighting conditions (shape numScenes*numProbes x numProbes)
    '''
    dataset_dir = './input/kinect'
    img_names = [name for name in os.listdir(dataset_dir)]
    img_list = []
    lights_list = []
    for img in img_names:
        
        img_filenames = [os.path.join(dataset_dir,img,name) for name in os.listdir(os.path.join(dataset_dir,img)) if 
            (name.endswith('.jpg') or name.endswith('.png')) and not (name.endswith('thumb.jpg') or name.endswith('materials_mip2.png') or name[:7]=='normals')
        ]
        
        img_list.append(img_filenames)
        for img_file in img_filenames:
            assert os.path.isfile(lights[0]+'_gray256.jpg')
            lights_list.append(lights)
    frames_list = [item for sublist in img_list for item in sublist]
    frames_list.sort()
    return frames_list, lights_list

def cropImage(img,w,h):
    h_img, w_img, _ = img.shape
        
    desired_aspect_ratio = w/h
    aspect_ratio = w_img/h_img

    if aspect_ratio<desired_aspect_ratio:
        desired_width = w_img*aspect_ratio/desired_aspect_ratio
    else:
        desired_width = w_img/aspect_ratio*desired_aspect_ratio
    wc = desired_width/2
    img_crop = img[:,int(w_img/2-wc):int(w_img/2+wc)]
    img_crop = cv2.resize(img_crop,(w,h))
    return img_crop

def fillBorder(img):
    borderSize = 11
    img[0:borderSize,:] = img[borderSize,:]
    img[:,0:borderSize] = img[:,borderSize].reshape((-1,1,3))
    img[-borderSize:,:] = img[-borderSize,:]
    img[:,-borderSize:,:] = img[:,-borderSize].reshape((-1,1,3))
    return img

def loadImg(path):
    '''
    Support for multiple file types
    '''
    if os.path.isfile(path):
        img_out = cv2.imread(path)
    else:
        print(path.split('.')[-2]+'.jpg')
        img_out = cv2.imread(path.split('.')[-2]+'.jpg')
    return img_out

if __name__ == "__main__":
    decompNet, normalNet = initNetworks()

    """
    Open the video file and start frame-by-frame processing
    """
    lights = ['input/lights/dir_0']
    #lights = ['input/lights/light_00','input/lights/light_01','input/lights/light_02','input/lights/light_03','input/lights/light_04','input/lights/light_05','input/lights/light_06']
    if opt.kinect:
        frame_list, lights_list = loadKinect(lights)
    elif opt.benchmark:
        frame_list, lights_list = loadDataset()
    elif opt.image is not None:
        frame_list = [os.path.join('input','images',opt.image)]
        lights_list = [lights]
    else:
        outputDir = os.path.join('output','videos')
        outputs = []
        streams = []
        for i in range(len(lights)):
            output = av.open(os.path.join(outputDir,opt.output_file.split('.')[-2]+lights[i].split('_')[-1]+'.mp4'),'w')
            stream = output.add_stream('mpeg4',24)
            stream.bit_rate = 8000000
            outputs.append(output)
            streams.append(stream)
        video = av.open('./input/videos/'+opt.video_name)
        frame_list = video.decode(video=0)

        lights_list = [] #(shape numScenes x numProbes)
        for i in range(video.streams.video[0].frames):
            lights_list.append(lights) 

    k = 0
    loss = []
    normals_list = []
    for frame in frame_list:
        if opt.benchmark or (opt.image is not None):
            img = loadImg(frame)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            scene, img_name = frame.split('/')[-2:]
            img_name = img_name.split('.')[-2]
            if opt.image is None and not opt.kinect:
                light_num = img_name.split('_')[-2]
        else:
            img = frame.to_image()
            img = np.asarray(img)
        if opt.kinect:
            img = np.uint8(np.clip(1.6*img, 0, 255)) #brighten kinect images
        
        img = cropImage(img,640,480)
        
        """
        Calculate albedo, shading and normals
        """

        albedo,shading_gt = get_decomposition(img,decompNet)
        shading_3 = np.zeros_like(albedo)
        for i in range(3):
            shading_3[:,:,i] = shading_gt

        if opt.gt_normals:
            no_ext = frame.split('.')[-2]
            name = no_ext.rstrip('0123456789')
            img_num = no_ext[len(name):]
            normal_path = os.path.join(*no_ext.split('/')[:-1])
            normals = loadImg(normal_path+'/normals_vis'+img_num+'.png')
            normals = cv2.cvtColor(normals,cv2.COLOR_BGR2RGB)
            normals = cropImage(normals,640,480)
            if opt.kinect:
                normals = ndimage.uniform_filter(normals,size=5)
                normals_list.append(np.uint8(normals))
                normals = average_frames(normals_list[-5:])
            nrm1 = convertNormalsGT(np.uint8(normals))
        else:
            normals = 127.5*(get_normals(img,albedo.shape,normalNet)+1)
            normals = fillBorder(normals)
            if not (opt.benchmark or opt.image is not None):
                normals_list.append(np.uint8(normals))
                normals = average_frames(normals_list[-3:])
            nrm1 = convertNormalsNew(normals)
        nrm1 = nrm1 / np.linalg.norm(nrm1, axis=2, keepdims=True)
        
        #normals = cropImage(normals)

        """
        Relight Image
        """
        
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

        for i,light in enumerate(lights_list[k]):
            if opt.benchmark and not opt.kinect and light.split('_')[-1] != light_num:
                continue #only run on matching lights
            diff = cv2.imread("%s_gray256.jpg" % (light))
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

            shading_pre, diff_coverage = relight(albedo, diff, nrm1, K_apprx)
            shading = recolor(shading_pre, shading_3)

            relit_diff = (shading/255)*albedo
            relit_diff = recolor_normalize(relit_diff,img)

            if opt.kinect:
                loss.append(SiMSE(relit_diff[135:340,135:480],img[135:340,135:480]))
            else:
                loss.append(SiMSE(relit_diff,img))
            print("Frame:",k,"  Light:",light.split('_')[-1],"  Si-MSE:",round(loss[-1],5),"(current)",round(np.mean(loss),5),"(average)")
            
            if opt.visualize:
                visualize(albedo,shading_gt,nrm1,shading_pre,shading,relit_diff)

            '''
            save image/video frame
            '''
            if opt.kinect:
                path = os.path.join('output','kinect',scene)
                if not os.path.exists(path):
                    os.makedirs(path)
                relit_diff = cv2.cvtColor(relit_diff,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path,img_name+'.jpg'),relit_diff)
            elif opt.benchmark:
                path = os.path.join('output','multi_dataset',scene)
                if not os.path.exists(path):
                    os.makedirs(path)
                relit_diff = cv2.cvtColor(relit_diff,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path,light.split('/')[-1]+'.jpg'),relit_diff)

            elif opt.image is not None:
                path = os.path.join('output','images',img_name)
                if not os.path.exists(path):
                    os.makedirs(path)
                relit_diff = cv2.cvtColor(relit_diff,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path,light.split('/')[-1]+'.jpg'),relit_diff)

            else:
                frame = av.VideoFrame.from_ndarray(relit_diff, format='rgb24')
                packet = streams[i].encode(frame)
                outputs[i].mux(packet)
        k += 1
    
    if not (opt.benchmark or opt.image is not None):
        for output in outputs:
            output.close()
    