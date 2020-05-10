import os
import h5py
import numpy as np
import requests
import argparse
import scipy.io
import zipfile
import shutil

from tqdm import tqdm
from pdb import set_trace
from PIL import Image

class PrepareDataset():
	def __init__(self, args):
		labeled_download_link = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
		filename = "nyu_depth_v2_labeled.mat"
		self.filepath = f"downloads/{filename}"
		self.mode = args.mode
		self.check_folder("downloads")
		if filename not in os.listdir("downloads"):
			print('[*] Downloading NYU Depth Dataset V2 labled(size: 2.8G)...')
			wget.download(labeled_download_link, self.filepath)
		# google drive file-id, compiled raw training data
		self.file_id = "10uDnRMpvP9epXk4BU2TyQ8SerwpGzOOB"

		# creating folders
		self.train_folder = "datasets/nyu_depth_v2/train"
		self.test_folder = "datasets/nyu_depth_v2/test"
		self.remove_folder(self.train_folder)
		self.remove_folder(self.test_folder)
		self.check_folder(self.train_folder)
		self.check_folder(self.test_folder)

	def check_folder(self, log_dir):
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
			return log_dir

	def remove_folder(self, log_dir):
		if os.path.exists(log_dir):
			shutil.rmtree(log_dir)

	def create_dataset(self):
		# dynamic method calling
		getattr(self, f"create_{self.mode}_dataset")()

	def create_raw_dataset(self):
		# download and place the training folder.
		print('[*] Downloading NYU Depth Dataset V2 Raw(size: 4.0G)...')
		download_path = "downloads/train.zip"
		self.download_file_from_google_drive(self.file_id, download_path)

		print('[*] Extracting NYU Depth Dataset training dataset...')
		with zipfile.ZipFile(download_path, 'r') as zip_ref:
			for file in tqdm(iterable = zip_ref.namelist(), total = len(zip_ref.namelist())):	
				zip_ref.extract(member = file, path = "datasets/nyu_depth_v2")

		print('[*] Preparing official NYU Depth test dataset...')
		# creating test data from official training/test split
		with h5py.File(self.filepath, 'r') as complete_data:

			train_test = scipy.io.loadmat("scripts/splits.mat")
			test_idx = set([int(x) for x in train_test["testNdxs"]])

			images = complete_data["images"]
			depths = complete_data["depths"]
			scenes = [u''.join(chr(c) for c in complete_data[obj_ref]) for obj_ref in complete_data['sceneTypes'][0]]

			for index, (image, depth, scene) in tqdm(enumerate(zip(images, depths, scenes))):
				if index in test_idx:
					self.check_folder(self.test_folder + "/" + scene)

					image = Image.fromarray(np.transpose(image, (2, 1, 0)))
					depth = Image.fromarray(np.transpose(depth, (1, 0)))
					depth = depth.convert("L")

					image.save(self.test_folder + "/" + scene + f"/rgb_{str(index).zfill(5)}.jpg")
					depth.save(self.test_folder + "/" + scene + f"/depth_{str(index).zfill(5)}.png")
			test_count = len(test_idx)
		
		print(f"No of Train image: 24231")
		print(f"No of Test image: {test_count}")
		print("[*] Done.")

	def create_light_dataset(self):
		with h5py.File(self.filepath, 'r') as complete_data:
			print("[*] processing dataset...")

			images = complete_data["images"]
			depths = complete_data["depths"]
			scenes = [u''.join(chr(c) for c in complete_data[obj_ref]) for obj_ref in complete_data['sceneTypes'][0]]
			scene_count = {}
			train_count = 0
			test_count = 0

			for index, (image, depth, scene) in tqdm(enumerate(zip(images, depths, scenes))):
				
				self.check_folder(self.train_folder + "/" + scene)
				self.check_folder(self.test_folder + "/" + scene)

				image = Image.fromarray(np.transpose(image, (2, 1, 0)))
				depth = Image.fromarray(np.transpose(depth, (1, 0)))
				depth = depth.convert("L")
				
				if scene in scene_count:
					scene_count[scene] += 1
					if scene_count[scene]%10 == 1:  # approximately 10% data for test set
						folder = self.test_folder
						test_count += 1
					else:
						folder = self.train_folder
						train_count += 1
				else:
					scene_count[scene] = 1
					folder = self.test_folder
					test_count += 1

				image.save(folder + "/" + scene + f"/rgb_{str(index).zfill(5)}.jpg")
				depth.save(folder + "/" + scene + f"/depth_{str(index).zfill(5)}.png")
			print(f"No of Train image: {train_count}")
			print(f"No of Test image: {test_count}")
		print("[*] Done.")


	# This function is copied from the source
	# Source: https://stackoverflow.com/a/39225039
	def download_file_from_google_drive(self, id, destination):
		def get_confirm_token(response):
			for key, value in response.cookies.items():
				if key.startswith('download_warning'):
					return value
			return None

		def save_response_content(response, destination):
			CHUNK_SIZE = 32768

			with open(destination, "wb") as f:
				for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
					if chunk: # filter out keep-alive new chunks
						f.write(chunk)

		URL = "https://docs.google.com/uc?export=download"

		session = requests.Session()

		response = session.get(URL, params = { 'id' : id }, stream = True)
		token = get_confirm_token(response)

		if token:
			params = { 'id' : id, 'confirm' : token }
			response = session.get(URL, params = params, stream = True)

		save_response_content(response, destination)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Prepare training dataset from the NYU Depth V2 dataset')
	parser.add_argument('--mode', type = str, default = "raw", choices=['light', 'raw'], help = "Select training mode for the task")
	args = parser.parse_args()
	creator = PrepareDataset(args)
	creator.create_dataset()
	print("You can delete files from the downloads directory to free up the extra space.")
