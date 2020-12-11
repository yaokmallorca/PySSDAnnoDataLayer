
# coding: utf-8
"""VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot
"""
import os
import os.path as osp
import sys
caffe_root='/home/yaok/software/caffe_ssd/'
os.chdir(caffe_root)
sys.path.append(caffe_root+'python')
import caffe
import PIL.Image as Image
from random import shuffle

import cv2
import numpy as np
import scipy
from augmentation import SSDAugmentation
from xml.dom import minidom
if sys.version_info[0] == 2:
	import xml.etree.cElementTree as ET
else:
	import xml.etree.ElementTree as ET

try:
	xrange
except:
	xrange = range


class PyAnnotatedDataLayer(caffe.Layer):

	"""
	This is a simple synchronous datalayer for training a multilabel model on
	PASCAL.
	"""

	def setup(self, bottom, top):

		self.top_names = ['data', 'label']
		# === Read input parameters ===
		# params is a python dictionary with layer parameters.
		self.params = eval(self.param_str)
		# Check the parameters for validity.
		# check_params(params)
		# store input as class variables
		self.batch_size = self.params['batch_size']
		# Create a batch loader to load the images.
		self.batch_loader = BatchLoader(self.params, None)
		# === reshape tops ===
		# since we use a fixed input image size, we can shape the data layer
		# once. Else, we'd have to do it in the reshape call.
		top[0].reshape(
		    self.batch_size, 3, self.params['im_shape'][0], self.params['im_shape'][1])
		# Note the 20 channels (because PASCAL has 20 classes.)
		top[1].reshape(1, 1, 1, 8)
		# print_info("PascalDataLayer", params)

	def forward(self, bottom, top):
		"""
		Load data.
		"""
		num_bboxes = 0
		imgs = np.zeros((self.batch_size, 3, 
						self.params['im_shape'][0], 
						self.params['im_shape'][1]))
		labels = np.array([])
		for itt in range(self.batch_size):
			im, label = self.batch_loader.load_next_image(itt)
			im = im.transpose((2,0,1))
			num_bboxes += label.shape[0]
			imgs[itt] = im
			if labels.shape[0] == 0:
				labels = label.copy()
			else:
				labels = np.vstack((labels, label))

		top[1].reshape(1, 1, num_bboxes, 8)
		top[0].data[...] = imgs
		top[1].data[0,0,...] = labels

		"""
		for itt in range(self.batch_size):
			# Use the batch loader to load the next image.
			im, label = self.batch_loader.load_next_image(itt)
			im = im.transpose((2,0,1))
			# num_bboxes = max(label.shape[0], 1)
			num_bboxes = len(label)
			top[1].reshape(1, 1, num_bboxes, 8)
			# Add directly to the caffe data layer
			# print("################################")
			top[0].data[itt, ...] = im
			top[1].data[0, 0,...] = label
		print("num_bboxes: ", num_bboxes)
		"""

	def reshape(self, bottom, top):
		"""
		There is no need to reshape the data, since the input is of fixed size
		(rows and columns)
		"""
		pass

	def backward(self, top, propagate_down, bottom):
		"""
		These layers does not back propagate
		"""
		pass


class BatchLoader(object):

	"""
	This class abstracts away the loading of images.
	Images can either be loaded singly, or in a batch. The latter is used for
	the asyncronous data layer to preload batches while other processing is
	performed.
	"""

	def __init__(self, params, result):
		self.result = result
		self.batch_size = params['batch_size']
		self.pascal_root = params['sbdd_dir']
		self.im_shape = params['im_shape']
		self.mean = np.array(params['mean'])
		# get list of image indexes.
		list_file = params['split'] + '.txt'
		self.indexlist = [line.rstrip('\n') for line in open(
		    osp.join(self.pascal_root, 'ImageSets/Main', list_file))]
		self._cur = 0  # current image
		# this class does some simple data-manipulations
		self.transformer = SSDAugmentation(mean = self.mean)

		print("BatchLoader initialized with {} images".format(
			len(self.indexlist)))

	def load_next_image(self, item_id):
		"""
		Load the next image in a batch.
		"""
		# Did we finish an epoch?
		if self._cur == len(self.indexlist):
			self._cur = 0
			shuffle(self.indexlist)

		# Load an image
		index = self.indexlist[self._cur]  # Get the image index
		image_file_name = index + '.png'
		im = np.asarray(Image.open(
			osp.join(self.pascal_root, 'JPEGImages', image_file_name)))
		im = scipy.misc.imresize(im, self.im_shape)  # resize

		# do a simple horizontal flip as data augmentation
		# flip = np.random.choice(2)*2-1
		# im = im[:, ::flip, :]

		# Load and prepare ground truth
		# label = np.zeros(7).astype(np.float32)
		anns = load_pascal_annotation(index, self.pascal_root)

		self._cur += 1
		im_aug, anns['boxes'], anns['gt_classes'] = self.transformer(im, 
				anns['boxes'], anns['gt_classes'])

		labels = []
		i=0
		for box, label in zip(anns['boxes'], anns['gt_classes']):
			labels.append([item_id, label, i, box[0], box[1], box[2], box[3], 0])
			# print(item_id, label, i, box[0], box[1], box[2], box[3], 0)
			i+=1
		labels = np.array(labels) #.astype(np.float32)
		# return self.transformer(im, anns['boxes'], anns['gt_classes'])
		return im_aug, labels


def load_pascal_annotation(index, pascal_root):
	"""
	This code is borrowed from Ross Girshick's FAST-RCNN code
	(https://github.com/rbgirshick/fast-rcnn).
	It parses the PASCAL .xml metadata files.
	See publication for further details: (http://arxiv.org/abs/1504.08083).
	Thanks Ross!
	"""
	classes = ( '_background',
				'label', 'blue paper tape', 'pink paper tape', 
				'black paper tape', 'lock', 'occupy_filter', 
				'empty_filter')
	class_to_ind = dict(zip(classes, xrange(len(classes))))

	filename = osp.join(pascal_root, 'Annotations', index + '.xml')
	# print 'Loading: {}'.format(filename)

	def get_data_from_tag(node, tag):
		return node.getElementsByTagName(tag)[0].childNodes[0].data

	with open(filename) as f:
		data = minidom.parseString(f.read())

	sizes = data.getElementsByTagName('size')
	for size in enumerate(sizes):
		width = float(get_data_from_tag(size[1], 'width'))
		height = float(get_data_from_tag(size[1], 'height'))

	objs = data.getElementsByTagName('object')
	num_objs = len(objs)

	boxes = np.zeros((num_objs, 4), dtype=np.float32)
	gt_classes = np.zeros((num_objs), dtype=np.int32)
	overlaps = np.zeros((num_objs, 21), dtype=np.float32)

	# Load object bounding boxes into a data frame.
	for ix, obj in enumerate(objs):
		# Make pixel indexes 0-based
		xmin = float(get_data_from_tag(obj, 'xmin')) # - 1
		ymin = float(get_data_from_tag(obj, 'ymin')) # - 1
		xmax = float(get_data_from_tag(obj, 'xmax')) # - 1
		ymax = float(get_data_from_tag(obj, 'ymax')) # - 1
		if xmin > width or xmin < 0:
			print("bounding box exceeds image boundary: xmin: ", xmin, " width: ", width)
		if xmax > width or xmax < 0:
			print("bounding box exceeds image boundary: xmax: ", xmax, " width: ", width)
		if ymin > width or ymin < 0:
			print("bounding box exceeds image boundary: ymin: ", ymin, " width: ", width)
		if ymax > width or ymax < 0:
			print("bounding box exceeds image boundary: ymax: ", ymax, " width: ", width)
		if xmin >= xmax or ymin >= ymax:
			print("bounding box irregular: xmin: ", xmin, " xmax: ", xmax, " ymin: ", ymin, " ymax: ", ymax)
		xmin = xmin/width
		xmax = xmax/height
		ymin = ymin/width
		ymax = ymax/height
		cls = class_to_ind[
			str(get_data_from_tag(obj, "name")).lower().strip()]
		boxes[ix, :] = [xmin, ymin, xmax, ymax]
		gt_classes[ix] = cls
		overlaps[ix, cls] = 1.0
	overlaps = scipy.sparse.csr_matrix(overlaps)

	return {'boxes': boxes,
			'gt_classes': gt_classes,
			'gt_overlaps': overlaps,
			'flipped': False,
			'index': index}

if __name__ == '__main__':
	name = "223312505"
	pascal_root = "/media/data/seg_dataset/fbox/"
	img_path = os.path.join(pascal_root, 'JPEGImages', name+'.png')
	img = cv2.imread(img_path)

	aug_ssd = SSDAugmentation(mean=[132.95022975, 132.9561738, 130.17624285])
	det_label = load_pascal_annotation(name, pascal_root)
	print('det_label: ', det_label)

	aug_img, aug_box, aug_label = aug_ssd(img, det_label['boxes'], det_label['gt_classes'])
