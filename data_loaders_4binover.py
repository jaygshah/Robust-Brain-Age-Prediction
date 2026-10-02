from collections import Counter
from glob import glob
from torch.utils import data
from monai import transforms as T
import nibabel as nib
import numpy as np
import os
import torch

# Age groups used for stratified oversampling: 8-12, 12-16, ..., 92-96
AGE_BIN_EDGES = np.arange(8, 97, 4)


def list_age_folders(image_dir):
	"""Return the ages (as ints) of the per-age subfolders in image_dir, in numeric order."""
	ages = []
	for path in glob(os.path.join(image_dir, '*')):
		name = os.path.basename(path)
		if os.path.isdir(path) and name.isdigit():
			ages.append(int(name))
	return sorted(ages)


def age_bin(age):
	return int(np.clip(np.digitize(age, AGE_BIN_EDGES) - 1, 0, len(AGE_BIN_EDGES) - 2))


class DataFolder(data.Dataset):
	def __init__(self, image_dir, image_type, transform, mode, min_age, max_age, keep_out_of_range=False):
		# Dictionary to map image types to their respective loading functions
		self.__image_reader = {
			'np': lambda url: np.load(url),
			'nii': lambda url: nib.load(url).get_fdata()
		}
		self.__extensions = {
			'np': ['*.npy'],
			'nii': ['*.nii', '*.nii.gz'],
		}

		assert image_type in self.__image_reader
		assert transform is not None

		self.image_dir = image_dir
		self.image_type = image_type
		self.transform = transform
		self.mode = mode
		self.min_age = min_age
		self.max_age = max_age
		self.keep_out_of_range = keep_out_of_range
		self.num_classes = max_age - min_age + 1
		self.samples = []

		self.__process()

		print("Total data loaded =", len(self.samples))

	def __process(self):
		files_per_age = {}
		for age in list_age_folders(self.image_dir):
			# Out-of-range labels are invalid class targets, so they are only usable for inference
			if not self.keep_out_of_range and not self.min_age <= age <= self.max_age:
				print(f"Skipping age folder {age}: outside the model's age range [{self.min_age}, {self.max_age}]")
				continue
			files = []
			for pattern in self.__extensions[self.image_type]:
				files += glob(os.path.join(self.image_dir, str(age), pattern))
			files_per_age[age] = sorted(files)
		print("Ages: ", len(files_per_age))

		# Oversample under-represented age groups in the training set
		sample_mul = {age: 1 for age in files_per_age}
		if self.mode == "train":
			bin_counts = Counter()
			for age, files in files_per_age.items():
				bin_counts[age_bin(age)] += len(files)
			maxval = max(bin_counts.values())
			for age in files_per_age:
				sample_mul[age] = maxval // bin_counts[age_bin(age)]
		print("Sample multiplier:", sample_mul)

		for age, files in files_per_age.items():
			label = age - self.min_age
			self.samples += [(url, label) for url in files] * sample_mul[age]

		assert len(self) > 0

	def __read(self, url):
		# Read the image from the URL using the appropriate reader
		return self.__image_reader[self.image_type](url)

	def __getitem__(self, index):
		url, lbl = self.samples[index]
		img = self.__read(url)

		# Apply transforms to the image and rescale intensities to [0, 1]
		img = np.expand_dims(img, 0)
		img = torch.as_tensor(self.transform(img), dtype=torch.float32)
		img = img - img.min()
		img_max = img.max()
		if img_max > 0:
			img = img / img_max
		return img, torch.LongTensor([lbl]), url

	def __len__(self):
		return len(self.samples)


def get_loader(image_dir, min_age, max_age, batch_size=5, mode='train', augment=True, num_workers=16,
			   keep_out_of_range=False):
	"""Build and return a data loader. Labels are class indices: age - min_age."""
	transform = []

	if mode == 'train' and augment:
		transform.append(T.RandGaussianNoise())
		transform.append(T.RandBiasField())
		transform.append(T.RandScaleIntensity(0.25))
		transform.append(T.RandAdjustContrast())
		transform.append(T.RandGibbsNoise())
		transform.append(T.RandKSpaceSpikeNoise())
		transform.append(T.RandRotate())
		transform.append(T.RandFlip())

	transform.append(T.ToTensor(dtype=torch.float, track_meta=False))
	transform = T.Compose(transform)

	dataset = DataFolder(image_dir, 'nii', transform, mode, min_age, max_age, keep_out_of_range)

	# Training batches must be full (ORDER loss needs at least 2 samples); evaluation keeps every
	# subject in a fixed order.
	is_train = mode == 'train'
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=is_train,
								  drop_last=is_train,
								  num_workers=num_workers,
								  pin_memory=torch.cuda.is_available())
	return data_loader


if __name__ == '__main__':
	ages = list_age_folders('./data/HC/train')
	loader = get_loader('./data/HC/train', min(ages), max(ages), mode="train")
	for i, x in enumerate(loader):
		print(i, x[0].shape, x[1], torch.min(x[0]), torch.max(x[0]))
		break
