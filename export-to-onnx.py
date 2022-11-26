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

#export to onnx
size = (512, 512)
t1 = torch.randn(1, 3, size[0], size[1], device='cuda')
t2 = torch.randn(1, 3, size[0], size[1], device='cuda')

torch.onnx.export(model.module,      
	                   (t1, t2),
	                   f'fast_acvnet_{size[0]}_{size[1]}.onnx',
	                   export_params=True,
	                   opset_version=16,
	                   do_constant_folding=True,  
	                   input_names = ['left', 'right'],
	                   output_names = ['output'])