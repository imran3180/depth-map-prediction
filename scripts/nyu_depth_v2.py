import os
import wget
import zipfile
import scipy.io as io
import h5py
import numpy as np
import torch

from tqdm import tqdm
from random import shuffle
from pdb import set_trace
import datetime
import time

download_link = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
filename = "scripts/nyu_depth_v2_labeled.mat"

# wget.download(download_link, filename)
with h5py.File(filename, 'r') as complete_data:
	print("processing dataset...")
	data = []
	for index, (image, depth) in tqdm(enumerate(zip(complete_data["images"], zip(complete_data["depths"])))):
		data.append(
			{
				"index": index,
				"rgb":  torch.from_numpy(np.transpose(image, (0, 2, 1))),
				"depth": torch.from_numpy(np.transpose(depth[0], (1, 0)))
			}
		)
	shuffle(data)						# random shuffle the data
	train = data[:-256]			# 1193 images
	val = data[-256:-128]		#	128 images
	test = data[-128:]			# 128 images

def check_folder(log_dir):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  return log_dir

check_folder("datasets/nyu_depth_v2")
torch.save(train, "datasets/nyu_depth_v2/train.pt")
torch.save(val, "datasets/nyu_depth_v2/val.pt")
torch.save(test, "datasets/nyu_depth_v2/test.pt")