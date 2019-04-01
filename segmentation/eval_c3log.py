import numpy as np
import torch
import glob

import cv2
import os
from argparse import ArgumentParser
from cnn import SegmentationModel as net
from torch import nn
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

pallete = [[0, 0, 0],
           [128, 64, 128]]



def evaluateModel(args, model, image_list):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    model.eval()
    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName)
        if args.overlay:
            img_orig = np.copy(img)

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        # resize the image to 1024x512x3
        img = cv2.resize(img, (args.inWidth, args.inHeight))
        if args.overlay:
            img_orig = cv2.resize(img_orig, (args.inWidth, args.inHeight))

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        if args.gpu:
            img_tensor = img_tensor.cuda()
        img_out = model(img_tensor)

        classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        # upsample the feature maps to the same size as the input image using Nearest neighbour interpolation
        # upsample the feature map from 1024x512 to 2048x1024

        print(np.unique(classMap_numpy))

        classMap_numpy = cv2.resize(classMap_numpy, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST)
        if i % 100 == 0 and i > 0:
            print('Processed [{}/{}]'.format(i, len(image_list)))

        name = imgName.split('/')[-1]

        if args.colored:
            classMap_numpy_color = np.zeros((img.shape[1], img.shape[2], img.shape[0]), dtype=np.uint8)
            # for idx in range(len(pallete)):
            #     [r, g, b] = pallete[idx]
            #     classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            classMap_numpy_color[classMap_numpy == 1] = [128, 128, 128]
            #print(np.unique(classMap_numpy_color))

            cv2.imwrite(args.savedir + os.sep + 'c_' + name.replace(args.img_extn, 'png'), classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0)
                cv2.imwrite(args.savedir + os.sep + 'over_' + name.replace(args.img_extn, 'jpg'), overlayed)

        cv2.imwrite(args.savedir + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)
        sys.exit(0)


def main(args):
    # read all the images in the folder
    # image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)
    image_list = ["./data/179461a2-68e9-4a13-99e3-66f2f5904c7b_13922-mycrop.png"]
    modelA = net.EESPNet_Seg(args.classes, s=args.s)
    if not os.path.isfile(args.pretrained):
        print('Pre-trained model file does not exist. Please check ./pretrained_models folder')
        exit(-1)
    modelA = nn.DataParallel(modelA)
    modelA.load_state_dict(torch.load(args.pretrained))
    if args.gpu:
        modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)

    evaluateModel(args, modelA, image_list)




'''
CUDA_VISIBLE_DEVICES=0 \
python eval_c3log.py --s 1.0 --colored True --pretrained ./results_espnetv2_1/model_best.pth --data_dir /media/benw/Data/data/aws_log_data/train1k_800x800/val100_image

jetson:
python eval_c3log.py --s 1.0 --colored True --pretrained ./results_espnetv2_1/model_best.pth
'''
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--savedir', default='./results_c3log', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--pretrained', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=0.5, type=float, help='scale')

    parser.add_argument('--colored', default=False, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=False, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset. 20/35 for Cityscapes')

    args = parser.parse_args()
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    main(args)
