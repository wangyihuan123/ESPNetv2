import sys
import re
import os
import glob
import json
import collections
import cv2
import errno
import numpy as np
import shutil

'''
before run this script please exec:

cd /media/benw/Data/data/aws_log_data/data
ls -1 metadata/ | grep mta.json | head -n 25000 > ../trainval24k/trainvallist24k.txt

prepare the training images:
1. crop the log first 
2. change the relative coordinate of polygon
3. resize to width_res x height_res

hint: there are 565 small resolution images captured from depth-camera saved in small_vallist.txt
which mean, 24k is actually 23,379 images in trainval
'''

all_train_list = "/media/benw/Data/data/aws_log_data/train24k_833x833/trainlist24k.txt"
all_val_list = "/media/benw/Data/data/aws_log_data/vallist.txt"
width_res = 1024
height_res = 512

DATA_MAIN_DIR = "/media/benw/Data/data/aws_log_data/data"
DATA_LOG_DIR = "logdata"
DATA_META_DIR = "metadata"
DATA_TEST_LOG_DIR = "test_logdata"
DATA_TEST_META_DIR = "test_metadata"
METADATA_EXTN = "*.mta.json"
max_process_num = 1000

dataset_dir = "/media/benw/Data/data/aws_log_data/train1k_1024x512"
##### train ####
fname = all_train_list
sample_img_dst_dir = dataset_dir + "/train1k_image"
DATA_TRAINVAL_SEG_ANNO_DIR = "train1k_seganno"
seg_anno_dir = os.path.join(dataset_dir, DATA_TRAINVAL_SEG_ANNO_DIR)
meta_dir = os.path.join(DATA_MAIN_DIR, DATA_META_DIR)
img_dir = DATA_LOG_DIR

##### val #####
# fname = all_val_list
# sample_img_dst_dir = dataset_dir + "/val100_image"
# DATA_VAL_SEG_ANNO_DIR = "val100_seganno"
# seg_anno_dir = os.path.join(dataset_dir, DATA_VAL_SEG_ANNO_DIR)
# meta_dir = os.path.join(DATA_MAIN_DIR, DATA_TEST_META_DIR)
# img_dir = DATA_TEST_LOG_DIR



def crop_image(image, crop_bbox):
    return image[crop_bbox["y1"]:crop_bbox["y2"], crop_bbox["x1"]:crop_bbox["x2"]]


def generate_new_poylygonPoints(polygonPoints, crop_bbox):
    polygonPoints[:, 0] = polygonPoints[:,0] - crop_bbox["x1"]
    polygonPoints[:, 1] = polygonPoints[:,1] - crop_bbox["y1"]
    if (polygonPoints < 0).all() :
        return None

    return polygonPoints


def save_image(crop_image, jpg_file):

    basename = os.path.basename(jpg_file).split(".jpg") [0] + "-mycrop.png"
    filename = os.path.join(sample_img_dst_dir, basename)
    crop_image = cv2.resize(crop_image, (width_res, height_res))
    cv2.imwrite(filename, crop_image)

    return filename


# use mask? yes, just like Geoff's mask
def save_mask_as_annotation(image, jpg_file, polygonPoints):
    h, w = image.shape[:2]
    # print(h, w, jpg_file)

    # polygonPoints = mtaJson['polygonPoints']
    polygonPoints = [[int(p[0]), int(p[1])] for p in polygonPoints]

    mask = np.zeros((h, w), np.uint8)
    # cv2.drawContours(mask, np.array([polygonPoints]), 0, 255, -1)
    cv2.drawContours(mask, np.array([polygonPoints]), 0, 1, -1)

    # #######debug#########
    # alpha = 0.85
    # beta = 1-alpha
    # gamma = 0
    # img_add = cv2.addWeighted(image, alpha, mask, beta, gamma)
    # img_add = cv2.resize(img_add, (resize, resize))
    # cv2.imshow('img_add',img_add)
    # cv2.waitKey()
    # #####debug end#######

    # save the mask #
    if not os.path.exists(seg_anno_dir):
        try:
            os.makedirs(seg_anno_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(seg_anno_dir):
                pass
            else:
                raise
    annotation_file = os.path.join(seg_anno_dir, os.path.basename(jpg_file).split(".jpg")[0] + "-mycrop.png")
    # print(annotation_file)

    mask = cv2.resize(mask, (width_res, height_res))
    cv2.imwrite(annotation_file, mask)


def main():
    # tst_file = "dataset/c3b2befc-840c-48bd-ac98-76f66c23290a_1742-crop-3168818941894.jpg.mta.json"
    # with open(tst_file) as f:
    #     mtajson = json.load(f)

    #
    # img = cv2.imread("dataset/c3b2befc-840c-48bd-ac98-76f66c23290a_1742.jpg")
    # print(img.shape)
    # img = cv2.rectangle(img, (316, 88), (316+1894, 88+1894), (0,255,0),3)
    # img = cv2.resize(img, (720, 640))
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # return


    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    if not os.path.isdir(sample_img_dst_dir):
        os.makedirs(sample_img_dst_dir)

    trainval_list = fname

    with open(trainval_list) as f:
        mta_list = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    mta_list = [x.strip() for x in mta_list]
    filenames = []
    for mta_file in mta_list:
        mta_file = meta_dir + "/" + mta_file
        filenames.append(mta_file)

        with open(mta_file) as f:
            mtaJson = json.load(f)

        matches = re.search('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_[0-9]+',
                            mtaJson['image'])
        if matches is None:
            print("can't find image file based on mta.json: {}".format(mta_file))
            continue

        raw_file = matches.group(0) + ".jpg"
        jpg_file = os.path.join(DATA_MAIN_DIR, img_dir, raw_file)
        # print(jpg_file)

        if not os.path.exists(jpg_file):
            print("wrong jpg file", jpg_file)
            continue

        basename = os.path.basename(jpg_file).split(".jpg")[0] + "-mycrop.jpg"
        file_name = os.path.join(sample_img_dst_dir, basename)

        if os.path.exists(file_name):
            continue

        polygonPoints = np.array(mtaJson["polygonPoints"])
        polygonPoints = polygonPoints.astype(np.int)
        x_polygonPoints = polygonPoints[:, 0]
        y_polygonPoints = polygonPoints[:, 1]

        polygonPoints_top = np.amin(y_polygonPoints)
        polygonPoints_bottom = np.amax(y_polygonPoints)
        polygonPoints_right = np.amax(x_polygonPoints)
        polygonPoints_left = np.amin(x_polygonPoints)


        image = cv2.imread(jpg_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("failed to open image file " % jpg_file)
            continue

        image_height, image_width = image.shape


        crop_height = min((polygonPoints_bottom - polygonPoints_top) * 1.1, image_height)
        crop_width = min((polygonPoints_right - polygonPoints_left) * 1.1, image_width)

        width_padding_half = crop_width * 0.05
        height_padding_half = crop_height * 0.05


        crop_x1 = int( max(polygonPoints_left - width_padding_half, 0) )
        crop_x2 = int(crop_x1 + crop_width)
        crop_y1 = int( max(polygonPoints_top - height_padding_half, 0) )
        crop_y2 = int(crop_y1 + crop_height)

        # crop_x1 = int((image_width - crop_width) / 2.0)
        # crop_x2 = int(crop_x1 + crop_width)
        # crop_y1 = int((image_height - crop_height) / 2.0)
        # crop_y2 = int(crop_y1 + crop_height)

        crop_bbox = {"x1": crop_x1, "y1": crop_y1, "x2": crop_x2, "y2": crop_y2}

        # print(polygonPoints_left,polygonPoints_top,
        #       polygonPoints_right,polygonPoints_bottom)
        # print(crop_bbox)
        # print( image_width - crop_width - crop_x1, image_width - crop_x2)
        new_polygonPoints = generate_new_poylygonPoints(polygonPoints, crop_bbox)
        if new_polygonPoints is None:
            continue

        crop_img = crop_image(image, crop_bbox)

        # # debug
        # crop_img = cv2.resize(crop_img, (resolution, resolution))
        # cv2.imshow("crop_img", crop_img)
        # cv2.waitKey()
        #
        # showimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # showimage = cv2.rectangle(showimage, (polygonPoints_left, polygonPoints_top),
        #                           (polygonPoints_right, polygonPoints_bottom), (0, 255, 0), 3)
        # showimage = cv2.resize(showimage, (1024, 1024))
        # cv2.imshow("image", showimage)
        # cv2.waitKey()
        # #
        # # return

        crop_imagefile = save_image(crop_img, jpg_file)
        save_mask_as_annotation(crop_img,  jpg_file, new_polygonPoints)

        print("{}".format(len(filenames)))

        if len(filenames) >= max_process_num:
            break



if __name__ == '__main__':
    main()
