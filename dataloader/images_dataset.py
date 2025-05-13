from torch.utils.data import Dataset
from PIL import Image
from util import data_utils
import numpy as np
import random
import json
import os
import torch
from .mask_generator_256 import RandomMask
import re
import torchvision.transforms as transforms
from imageio import imread

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    items.sort(key=natural_keys)


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, use_mask=False, use_captions=False, return_name=False, mask_root=None, datamax=None, hole_range=[0,1], use_labels=False, z_dim=512):
		self.source_paths = data_utils.make_dataset(source_root, opts.check)
		self.target_paths = data_utils.make_dataset(target_root, opts.check)
		if mask_root is not None:
			self.mask_paths = data_utils.make_dataset(mask_root)
		else:
			self.mask_paths = None
		self.source_root = source_root
		self.target_root = target_root
		natural_sort(self.source_paths[0])
		natural_sort(self.source_paths[1])
		natural_sort(self.target_paths[0])
		natural_sort(self.target_paths[1])
		if self.mask_paths is not None:
			natural_sort(self.mask_paths[0])
			natural_sort(self.mask_paths[1])
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self._use_captions = use_captions
		self._use_mask = use_mask
		self.return_name = return_name
		self._raw_captions = None
		self._raw_texts = None
		self._len = len(self.source_paths[0])
		self.hole_range = hole_range
		self.datamax = datamax
		self._use_labels = use_labels
		self._raw_labels = None
		self.z_dim = z_dim

	def __len__(self):
		return self._len

	def resize(self, img, height, width, centerCrop=True):
		imgh, imgw = img.shape[0:2]

		if centerCrop and imgh != imgw:
			# center crop
			side = np.minimum(imgh, imgw)
			j = (imgh - side) // 2
			i = (imgw - side) // 2
			img = img[j:j + side, i:i + side, ...]

		# img = imresize(img, [height, width])
		img = np.array(Image.fromarray(img).resize((height, width)))
		return img	
	
	def __getitem__(self, index):
		from_path = self.source_paths[0][index]  # 全地址：root_dir + sub_dir
		from_path_rel = self.source_paths[1][index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[0][index]
		to_path_rel = self.target_paths[1][index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		img_size = to_im.size()[1:]

		if self._use_captions:
			text = self.get_text(index)
			token = torch.tensor(self.get_caption(index), dtype=torch.int)
		else:
			text = 'None'
			token = 'None'
		
		label = torch.from_numpy(self.get_label(index))
		
		if self._use_mask:
			if self.mask_paths is None:
				mask = RandomMask(img_size[-1], hole_range=self.hole_range)  # hole as 0, reserved as 1
				mask = np.ones(mask.shape) - mask  # hole as 1, reserved as 0
				mask = torch.tensor(mask, dtype=torch.float)
			else:
				mask_path = self.mask_paths[0][index]
				mask = imread(mask_path)
				mask = self.resize(mask, self.opts.fineSize, self.opts.fineSize)
				if len(mask.shape) < 3:
					mask = mask[..., np.newaxis]
					mask = np.tile(mask, (1,1,3))
				else:
					mask = mask[:, :, 0]
					mask = mask[..., np.newaxis]
					mask = np.tile(mask, (1,1,3))
				mask = transforms.ToTensor()(mask)
				mask = mask[0]
				mask = mask.unsqueeze(0)
		else:
			mask = 'None'

		if self.return_name:
			name = os.path.basename(to_path)
		else:
			name = 'None'

		z = torch.randn([self.z_dim])

		data = {
			'name': name,
			'mask': mask,
			'from_im': from_im,
			'to_im': to_im,
			'text': text,
			'token': token,
			'label': label,
			'z': z,
		}

		return data

	def _load_raw_captions(self):  # 重写。先将coco数据集重新整理一个json文件
        # 从每个图片的captions里随机选一个
		fname = 'dataset.json'
		if not os.path.exists(os.path.join(self.target_root, fname)):
			return None
		with open(os.path.join(self.target_root, fname), 'rb') as f:
			captions = json.load(f)['labels']  # 获得的是一个list
		if captions is None:
			return None
		captions = dict(captions)
		output_texts = []
		output_tokens = []
		for fname in self.target_paths[1]:
			choices = captions[fname.replace('\\', '/')]
			choice = random.choice(choices)
			output_texts.append(choice)
			chosen_token = clip.tokenize(choice)  # [1, 77]  tensor
			output_tokens.append(chosen_token[0].numpy())
		output_tokens = np.array(output_tokens)
		return output_texts, output_tokens

	def get_caption(self, index):
		caption_embed = self._get_raw_captions()[index]
		return caption_embed.copy()

	def get_text(self, index):
		text = self._get_raw_texts()[index]
		return text

	def _get_raw_texts(self):
		if self._raw_texts is None:
			_raw_captions = self._load_raw_captions() if self._use_captions else None
			self._raw_texts = _raw_captions[0]
			self._raw_captions = _raw_captions[1]
			if self._raw_captions is None:
				self._raw_captions = np.zeros([self._len, 0], dtype=np.float32)
			assert isinstance(self._raw_captions, np.ndarray)
			assert self._raw_captions.shape[0] == self._len
		return self._raw_texts

	def _get_raw_captions(self):
		if self._raw_captions is None:
			_raw_captions = self._load_raw_captions() if self._use_captions else None
			self._raw_texts = _raw_captions[0]
			self._raw_captions = _raw_captions[1]
			if self._raw_captions is None:
				self._raw_captions = np.zeros([self._len, 0], dtype=np.float32)
			assert isinstance(self._raw_captions, np.ndarray)
			assert self._raw_captions.shape[0] == self._len
		return self._raw_captions

	def get_caption(self, index):
		caption_embed = self._get_raw_captions()[index]
		return caption_embed.copy()

	def get_label(self, index):
		label = self._get_raw_labels()[index]
		if label.dtype == np.int64:
			onehot = np.zeros(self.label_shape, dtype=np.float32)
			onehot[label] = 1
			label = onehot
		return label.copy()

	def _get_raw_labels(self):
		if self._raw_labels is None:
			self._raw_labels = self._load_raw_labels() if self._use_labels else None
			if self._raw_labels is None:
				self._raw_labels = np.zeros([self._len, 0], dtype=np.float32)
			assert isinstance(self._raw_labels, np.ndarray)
			assert self._raw_labels.shape[0] == self._len
			assert self._raw_labels.dtype in [np.float32, np.int64]
			if self._raw_labels.dtype == np.int64:
				assert self._raw_labels.ndim == 1
				assert np.all(self._raw_labels >= 0)
		return self._raw_labels

	def _load_raw_labels(self):
		fname = 'labels.json'
		if fname not in self._all_fnames:
			return None
		with self._open_file(fname) as f:
			labels = json.load(f)['labels']
		if labels is None:
			return None
		labels = dict(labels)
		labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
		labels = np.array(labels)
		labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
		return labels