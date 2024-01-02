from glob import glob
from torch.utils import data
from monai import transforms as T
import nibabel as nib
import numpy as np
import os
import random, csv
import torch
import pandas as pd

class DataFolder(data.Dataset):
	def __init__(self, image_dir, image_type, transform, mode='train'):
		self.__image_reader = {
			'np': lambda url: np.load(url),
			'nii': lambda url: nib.load(url).get_fdata()
		}

		self.__supported_extensions = self.__image_reader.keys()
		assert image_type in self.__supported_extensions
		assert transform != None

		self.image_dir = image_dir
		self.image_type = image_type
		self.transform = transform
		self.mode = mode
		self.data_urls = []
		self.data_labels = []
		self.data_index = []
		self.num_classes = 0

		self.__process()

		print("Total data loaded =", len(self.data_urls))

	def __process(self):

		classes = sorted([c.split("/")[-1] for c in glob(os.path.join(self.image_dir, '*'))])
		self.num_classes = len(classes)
		print("Classes: ", self.num_classes)

		df = pd.read_csv("masterdata.csv") 
		age = df['age'].tolist()

		hist, bin_edges = np.histogram(age, bins=[
			8, 12, 16, 20, 24, 28, 
			32, 36, 40, 44, 48, 
			52, 56, 60, 64, 68, 
			72, 76, 80, 84, 88, 92, 96])

		maxval = np.max(hist)
		multp = [int(np.floor( maxval/ freq)) for freq in hist]
		sample_mul = {}

		for i, c in enumerate(classes):
			for j in range(1, len(bin_edges)):
				
				if int(c)>=bin_edges[j-1] and int(c)<bin_edges[j]:
					if self.mode == "train":
						sample_mul[c] = multp[j-1]
					else:
						sample_mul[c] = 1

		print("Sample multiplier:", sample_mul)

		for i, c in enumerate(classes):
			temp = glob(os.path.join(self.image_dir, c, f'*.{self.image_type}'))
			self.data_urls += temp * sample_mul[c]
			self.data_labels += [i]*len(temp)*sample_mul[c]

		sorted(self.data_urls)

		self.data_index = list(range(len(self)))
		if self.mode in ['train']:
			random.seed(3141)
			random.shuffle(self.data_index)

		assert len(self) > 0

	def __read(self, url):
		return self.__image_reader[self.image_type](url)

	def __getitem__(self, index):
		img = self.__read(self.data_urls[self.data_index[index]])
		lbl = self.data_labels[self.data_index[index]]

		# # without transforms
		# img -= np.min(img)
		# img /= np.max(img)
		# return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor([lbl])

		# with transforms
		img = np.expand_dims(img, 0)
		img = self.transform(img)
		img -= np.min(img)
		img /= np.max(img)
		return torch.FloatTensor(img), torch.LongTensor([lbl])

	def __len__(self):
		return len(self.data_urls)


def get_loader(image_dir, crop_size=101, image_size=101, 
               batch_size=5, dataset='adni', mode='train', num_workers=16):
    """Build and return a data loader."""
    transform = []

    if mode == 'train':
    	transform.append(T.RandGaussianNoise())
    	transform.append(T.RandBiasField())
    	transform.append(T.RandScaleIntensity(0.25))
    	transform.append(T.RandAdjustContrast())
    	transform.append(T.RandGibbsNoise())
    	transform.append(T.RandKSpaceSpikeNoise())
    	transform.append(T.RandRotate())
    	transform.append(T.RandFlip())

    transform.append(T.ToTensor(dtype=torch.float))
    transform = T.Compose(transform)

    dataset = DataFolder(image_dir, 'nii', transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
	loader = get_loader('./data/adni/train', mode="train")
	for i, x in enumerate(loader):
		print(i, x[0].shape, x[1], torch.min(x[0]), torch.max(x[0]))
		break