import glob
import sys, os




# However, cmd is easier:
# ls -1 train24k_image > train.txt
# ls -1 val_image/  > val.txt



data_dir = ""
train_image_files = data_dir + "/train1k_image"
train_segmentannotation_files = data_dir + "/train1k_seganno"

train_txt = "train.txt"

if not os.path.exists(train_image_files):
    print("{} not exist".format(data_dir))
    sys.exit()

if not os.path.exists(train_segmentannotation_files):
    print("{} not exist".format(data_dir))
    sys.exit()


all_image_files = glob.glob(train_image_files + "/*.png")
all_seganno_files = glob.glob(train_segmentannotation_files + "/*.png")

if len(all_image_files) != len(all_seganno_files):
    print("len(all_image_files:{}) != len(all_seganno_files:{})".format(len(all_image_files), len(all_seganno_files)))
