import torch
from scipy.io import loadmat
from scipy.io import savemat
import scipy
import h5py
import numpy as np
import pdb
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import cv2

def image_stack(rgb_img, depth_image):
	fig=plt.figure(figsize=(12.8, 9.6))
	fig.add_subplot(1, 2, 1)
	plt.imshow(np.transpose(rgb_img, (2, 1, 0)))
	fig.add_subplot(1, 2, 2)
	plt.imshow(np.transpose(depth_image, (1, 0)))
	plt.show()

def transpose_image(image):
	if len(image.shape) == 2:
		return np.transpose(image, (1, 0))
	if len(image.shape) == 3:
		return np.transpose(image, (2, 1, 0))

def image_extension(image):
	if len(image.shape) == 2:
		return "png"
	if len(image.shape) == 3:
		return "jpg"

def imshow(img):
	pdb.set_trace()
	if len(img.shape) == 2:		# for depth image
		plt.imshow(np.transpose(img, (1, 0)))
	if len(img.shape) == 3:		# for rgb image 
		plt.imshow(np.transpose(img, (2, 1, 0)))

# Image: numpy ndarray(Shape: )
def resize_rgb_image(image, image_type, image_no, new_height, new_width):
	img = Image.fromarray(transpose_image(image))
	resized_img = transforms.functional.resize(img, (new_height, new_width))
	resized_img = np.array(resized_img)
	# return resized_img
	ext = image_extension(resized_img)
	resized_image = Image.fromarray(resized_img)
	# pdb.set_trace()
	# plt.savefig("../data/{}/{}".format(image_type, image_no))
	resized_image.save("data2/rgb/{}.{}".format(image_no, ext))

def resize_depth_image(image, image_type, image_no, new_height, new_width):
	img = Image.fromarray(transpose_image(image))
	resized_img = transforms.functional.resize(img, (new_height, new_width))
	resized_img = np.array(resized_img)
	ext = image_extension(resized_img)
	depth_image = ((resized_img - np.min(resized_img))/(np.max(resized_img) - np.min(resized_img))) * 255
	cv2.imwrite("data2/depth/{}.{}".format(image_no, ext), depth_image)

resized_image = np.ndarray(shape = (240, 320, 3, 1449))
resized_depth = np.ndarray(shape = (240, 320, 1449))

with h5py.File('../nyu_depth_v2_labeled.mat', 'r') as f:
	input_height = 240
	input_width = 320
	no_of_images = 1449

	images = f['images']
	depths = f['depths']

	for i in range(0, no_of_images):
		resize_rgb_image(images[i], "rgb", str(i).zfill(5), input_height, input_width)

	for i in range(0, no_of_images):
		resize_depth_image(depths[i], "depth", str(i).zfill(5), input_height, input_width)
	
	# for i in range(0, no_of_images):
	# 	resized_image[:,:,:,i] = resize_image(images[i], "image", i, input_height, input_width)
	# 	resized_depth[:,:,i] = resize_image(depths[i], "depth", i, input_height, input_width)
	
	# resized_data = {'image': resized_image, 'depth': resized_depth}
	# savemat('../resized_data', resized_data)
	




	# sample = {'image': images, 'depth': depths}
	# # resize image
	# resized_image=np.zeros((input_height, input_width, 3, no_of_images))
	# resized_depth=np.zeros((input_height, input_width, no_of_images))
	# for i in range(0,no_of_images):
	# 	resized_image[:,:,:,i] = scipy.misc.imresize(np.squeeze(sample['image'][:,:,:,i]),(input_height,input_width,3))
	# 	resized_depth[:,:,i] = scipy.misc.imresize(np.squeeze(sample['depth'][:,:,i]),(input_height,input_width))
	# 	print("total: {} images done.".format(i))
	# 	break

	# plt.subplot(121)
	# plt.imshow(sample['image'][:,:,:,0])
	# plt.title('original image')
	# plt.subplot(122)
	# plt.imshow(resized_image[:,:,:,0])
	# plt.title('resized image')
	# plt.show()



	# plt.subplot(121)
	# plt.imshow(sample['depth'][:,:,0])
	# plt.title('original depth')
	# plt.subplot(122)
	# plt.imshow(resized_depth[:,:,0])
	# plt.title('resized depth')
	# plt.show()
	# image_stack(resized_image[:,:,:,0], resized_depth[:,:,0])


	