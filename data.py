from __future__ import print_function
import zipfile
import os
import pdb

import torchvision.transforms as transforms

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)) # Calculate this statistics for the training image.
])

depth_data_transforms = transforms.Compose([
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)) # Calculate this statistics for the training image.
])

def initialize_data(folder):
    rgb_images = folder + '/rgb'
    if not os.path.isdir(rgb_images):
        raise(RuntimeError("Could not found {}/rgb folder".format(folder)))

    depth_images = folder + '/depth'
    if not os.path.isdir(depth_images):
        raise(RuntimeError("Could not found {}/depth folder".format(folder)))

    # Total Image - 1449 (Division: Trainging - 1024, Validation - 256, Testing - 169)
    dataset_prepared = True

    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        dataset_prepared = False
        os.mkdir(train_folder)
        os.mkdir(train_folder + '/rgb')
        os.mkdir(train_folder + '/depth')

    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        os.mkdir(val_folder)
        os.mkdir(val_folder + '/rgb')
        os.mkdir(val_folder + '/depth')

    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
        os.mkdir(test_folder + '/rgb')
        os.mkdir(test_folder + '/depth')

    if not dataset_prepared:
        for f in os.listdir(rgb_images):
            image_no = int(f.split(".")[0])
            if image_no < 1024:
                dest_folder = train_folder
            elif image_no < 1280:       # 1024 + 256
                dest_folder = val_folder
            else:
                dest_folder = test_folder
            os.rename(rgb_images + '/' + f, dest_folder + '/rgb' + '/' + f)
        for f in os.listdir(depth_images):
            image_no = int(f.split(".")[0])
            if image_no < 1024:
                dest_folder = train_folder
            elif image_no < 1280:       # 1024 + 256
                dest_folder = val_folder
            else:
                dest_folder = test_folder
            os.rename(depth_images + '/' + f, dest_folder + '/depth' + '/' + f)