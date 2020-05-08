import os
import sys
import h5py
import numpy as np
import torch
import requests

from tqdm import tqdm
from random import shuffle
from pdb import set_trace

# https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url, filename):
  with open(filename, 'wb') as f:
    response = requests.get(url, stream=True)
    total = response.headers.get('content-length')

    if total is None:
      f.write(response.content)
    else:
      downloaded = 0
      total = int(total)
      for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
        downloaded += len(data)
        f.write(data)
        done = int(100*downloaded/total)
        sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (100-done)))
        sys.stdout.flush()
  sys.stdout.write('\n')

def check_folder(log_dir):
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    return log_dir

download_link = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
filename = "nyu_depth_v2_labeled.mat"
filepath = f"downloads/{filename}"

check_folder("downloads")

if filename not in os.listdir("downloads"):
  print('[*] Downloading NYU Depth Dataset V2(size: 2.8G)...')
  download(download_link, filepath)

with h5py.File(filepath, 'r') as complete_data:
  print("[*] processing dataset...")
  data = []
  for index, (image, depth) in tqdm(enumerate(zip(complete_data["images"], zip(complete_data["depths"])))):
    data.append(
      {
        "index": index,
        "rgb":  torch.from_numpy(np.transpose(image, (0, 2, 1))),
        "depth": torch.from_numpy(np.transpose(depth[0], (1, 0)))
      }
    )
  shuffle(data)           # random shuffle the data
  train = data[:-256]     # 1193 images
  val = data[-256:-128]   # 128 images
  test = data[-128:]      # 128 images

check_folder("datasets/nyu_depth_v2")
torch.save(train, "datasets/nyu_depth_v2/train.pt")
torch.save(val, "datasets/nyu_depth_v2/val.pt")
torch.save(test, "datasets/nyu_depth_v2/test.pt")
print("[*] Done.")