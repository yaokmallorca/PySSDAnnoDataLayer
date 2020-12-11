#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
import vis
import matplotlib.pyplot as plt
import seaborn as sns
import random as rng

try:
    xrange
except:
    xrange = range

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


up_layers = ['conv4_3_l_c4', 'fc7_l_c3', 'conv6_2_c2', 'conv7_2_c1']

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.show()

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([119.916249, 113.8954083, 85.4776983])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        # detections = self.net.forward()['detection_out']
        self.net.forward()
        detections = self.net.blobs['detection_out'].data
        segmask = self.net.blobs['seg_score'].data[0]

        # print(detections.shape())
        # print(segmask.shape)
        # feat = self.net.blobs['conv4_3_l_c4'].data[0, :256]
        # print("feat shape: ", feat.shape)
        # feat = feat.sum(axis=0)
        # # vis_square(feat)
        # print("feat shape: ", feat.shape)
        # g = sns.heatmap(feat)
        # g.set(xticklabels=[])
        # g.set(yticklabels=[])
        # plt.show()
        # input('s')

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])

        hard_mask = np.argmax(segmask, axis=0)
        # cv2.imwrite("result.png", hard_mask*10)
        return result, hard_mask

    def seg2box(self, mask):
        mask_h, mask_w = mask.shape
        print(mask_h, mask_w, np.unique(mask))
        mask = mask.astype(np.uint8)

        ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("contours: ", len(contours))
        rect = []
        for i in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[i])
            xmin, ymin = float(x) / float(mask_w), float(y) / float(mask_h)
            xmax, ymax = float(x+w) / float(mask_w), float(y+h)/ float(mask_h)
            rect.append([xmin, ymin, xmax, ymax])

        return rect

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    img_dir = args.image_file
    img_path = '/media/data/seg_dataset/corrosion/JPEGImages'
    dst_path = '/home/yaok/software/caffe_ssd/result/corrosion/'

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    voc_palette = vis.make_palette(8)

    with open(img_dir) as f:
        img_list = f.readlines()

    for img_name in img_list:
        # img_name = 'IMG_20180305_120305'
        img_name = img_name.strip() + '.jpg'
        mask_name = img_name.strip()[0:-4] + '_seg.png'
        print(img_name)
        full_name = os.path.join(img_path, img_name)
        result, mask = detection.detect(full_name)

        mask_rect = detection.seg2box(mask)

        mask_h, mask_w = mask.shape
        print(result)
        img = cv2.imread(full_name)
        h, w, _ = img.shape
        img_mask = img.copy()
        mask_name = os.path.join(dst_path, mask_name)
        print(mask.shape, h, w)
        # cv2.resize(mask.astype(np.float32), (h, w), interpolation=cv2.INTER_CUBIC).astype(np.int32)
        img_mask = cv2.resize(img, (mask_h, mask_w), interpolation=cv2.INTER_CUBIC)

        out_im = vis.vis_seg(img_mask, mask, voc_palette)
        out_im = cv2.resize(out_im, (h, w), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(mask_name, out_im)
        # out_im.save(mask_name)
        # img = cv2.resize(out_im, (h, w), interpolation = cv2.INTER_CUBIC)
        det_bbox = []
        for item in result:
            xmin = int(round(item[0] * w))
            ymin = int(round(item[1] * h))
            xmax = int(round(item[2] * w))
            ymax = int(round(item[3] * h))
            det_bbox.append([xmin, ymin, xmax, ymax])
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            pt_x = int((xmax + xmin) / 2)
            pt_y = int((ymax + ymin) / 2)
            cv2.putText(img,item[-1] + str(item[-2]), 
                (pt_x, pt_y), 
                font, 
                fontScale,
                fontColor,
                lineType)
        det_bbox = np.array(det_bbox)
        print("det_bbox: ", det_bbox.shape, len(det_bbox))
        for box in mask_rect:
            # print(box)
            xmin = int(round(box[0] * w))
            ymin = int(round(box[1] * h))
            xmax = int(round(box[2] * w))
            ymax = int(round(box[3] * h))
            box_w = xmax - xmin
            box_h = ymax - ymin
            area = box_w * box_h
            if len(det_bbox) > 0:
                rect = np.array([xmin, ymin, xmax, ymax])
                ixmin = np.maximum(det_bbox[:, 0], rect[0])
                iymin = np.maximum(det_bbox[:, 1], rect[1])
                ixmax = np.minimum(det_bbox[:, 2], rect[2])
                iymax = np.minimum(det_bbox[:, 3], rect[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                uni = ((rect[2] - rect[0] + 1.) * (rect[3] - rect[1] + 1.) +
                       (det_bbox[:, 2] - det_bbox[:, 0] + 1.) *
                       (det_bbox[:, 3] - det_bbox[:, 1] + 1.) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                if ovmax == 0 and area > 300:
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            else:
                if area > 300:
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        dst_name = os.path.join(dst_path, img_name)
        cv2.imwrite(dst_name, img)


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',
                        default='/media/data/seg_dataset/corrosion/ImageSets/labelmap.prototxt')
    parser.add_argument('--model_def',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/corrosion/SSD_384x384/deploy.prototxt')
    parser.add_argument('--image_resize', default=384, type=int)
    parser.add_argument('--model_weights',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/corrosion/SSD_384x384/'
                        'VGG_corrosion_SSD_384x384_iter_19000.caffemodel') # 19000 25000
    parser.add_argument('--image_file', default='/media/data/seg_dataset/corrosion/ImageSets/Main/pics.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
