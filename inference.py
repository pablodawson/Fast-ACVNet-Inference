from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from datasets import __datasets__
from models import __models__
from utils import *
import skimage
import skimage.io
from PIL import Image
from datasets.data_io import get_transform

parser = argparse.ArgumentParser(description='Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACV)')
parser.add_argument('--left', type=str, default="input/left.png", help='input left image path')
parser.add_argument('--right', type=str, default="input/right.png", help='input right image path')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--output', type=str, default="out.png", help='output disparity path')
parser.add_argument('--loadckpt', default='pretrained/sceneflow.ckpt',help='load the weights from a specific checkpoint')

args = parser.parse_args()

model = __models__[args.model](args.maxdisp, False)
model = nn.DataParallel(model)
model.cuda()

#load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
model.eval()

processed = get_transform()

@make_nograd_func
def inference(imgL, imgR):
    
    left_img = processed(imgL)
    right_img = processed(imgR)
    
    # [w, h ,3 ] -> [1, w, h ,3]
    left_img = left_img[np.newaxis,:,:,:]
    right_img = right_img[np.newaxis,:,:,:]

    disp_ests = model(left_img.cuda(), right_img.cuda())
    disp_est = disp_ests[-1]

    disp_np = tensor2numpy(disp_est)

    disp_est_uint = np.round(disp_np * 256).astype(np.uint16)

    fn = "output.png"

    #[1, w, h ,3] -> [w, h ,3 ]
    disp_est_uint = np.stack((disp_est_uint,)*3, axis=-1)
    skimage.io.imsave(fn, disp_est_uint[0])

if __name__ == "__main__":
    imgL = Image.open(args.left).convert('RGB')
    imgR = Image.open(args.right).convert('RGB')

    #resize to power of 2
    imgL = imgL.resize((int(imgL.size[0]/32)*32, int(imgL.size[1]/32)*32), Image.BILINEAR)
    imgR = imgR.resize((int(imgR.size[0]/32)*32, int(imgR.size[1]/32)*32), Image.BILINEAR)

    inference(imgL, imgR)
