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
from shapely.geometry import Polygon

from voc_bbox_eval import *
from voc_seg_eval import *

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
categories = {'label'             : 1,
              'blue paper tape'   : 2, 
              'pink paper tape'   : 3, 
              'black paper tape'  : 4, 
              'lock'              : 5, 
              'occupy filter'     : 6, 
              'empty filter'      : 7}


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

def seg_inter_threshold(det, det_result, seg_result, n_cls, gamma):
    seg_mask_onehot = det.one_hot_transform(seg_result)
    seg_mask_onehot.transpose(1,2,0)
    thresh_inter = np.zeros(seg_result.shape).astype(np.uint8)
    seg_mask_onehot = seg_mask_onehot.astype(np.uint8)

    mask_h, mask_w = seg_result.shape
    mask_area = mask_h * mask_w
    det_mask = np.zeros((mask_h, mask_w, 3))
    for item in det_result:
        xmin = int(round(item[0] * mask_w))
        ymin = int(round(item[1] * mask_h))
        xmax = int(round(item[2] * mask_w))
        ymax = int(round(item[3] * mask_h))
        pt_x = int((xmax + xmin) / 2)
        pt_y = int((ymax + ymin) / 2)
        cls_ind = categories[item[-1]]
        for i in range(0, n_cls-1):
            if cls_ind == (i+1):
                ret, thresh = cv2.threshold(seg_mask_onehot[i], 0, 255, cv2.THRESH_BINARY)
                contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for j in range(len(contours)):
                    x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contours[j])
                    if w_contour * h_contour < 100:
                        continue
                    seg_roi_xmin, seg_roi_ymin = x_contour,             y_contour
                    seg_roi_xmax, seg_roi_ymax = x_contour+w_contour,   y_contour+h_contour
                    tmp_box_img = np.zeros((mask_h, mask_w, 3))
                    tmp_seg_img = np.zeros((mask_h, mask_w))
                    cv2.rectangle(tmp_box_img, (xmin, ymin), (xmax, ymax), (1,1,1), -1)
                    tmp_seg_img[seg_roi_ymin:seg_roi_ymax, seg_roi_xmin:seg_roi_xmax] = \
                        seg_mask_onehot[i, seg_roi_ymin:seg_roi_ymax, seg_roi_xmin:seg_roi_xmax]
                    tmp_inter = cv2.bitwise_and(tmp_box_img[:,:,0], tmp_seg_img)
                    tmp_area = float(len(np.where(tmp_inter == 1)[0])) / float(mask_area)
                    if tmp_area > gamma:
                        # thresh_inter[seg_roi_ymin:seg_roi_ymax, seg_roi_xmin:seg_roi_xmax] = \
                        #     tmp_seg_img[seg_roi_ymin:seg_roi_ymax, seg_roi_xmin:seg_roi_xmax]
                        thresh_inter[tmp_seg_img == 1] = cls_ind
                    del tmp_box_img
                    del tmp_seg_img
    return thresh_inter

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
        self.transformer.set_mean('data', np.array([132.95022975, 132.9561738, 130.17624285])) # mean pixel
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

        # display network shape 
        # for layer_name, blob in self.net.blobs.iteritems():
        #     print(layer_name + '\t' + str(blob.data.shape))
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
        return result, hard_mask

    # in_masks: n,1, h,w
    def one_hot_transform(self, in_masks, ignore=False, n_cl=8):
        one_hot = np.zeros((n_cl-1, in_masks.shape[0], in_masks.shape[1]))
        for i in range(1, n_cl):
            one_hot[i-1][in_masks == i] = 1
            # one_hot[i-1] = self.fillhole(one_hot[i-1])
        return one_hot

    def seg2box_multiclass(self, mask):
        one_hot_mask = self.one_hot_transform(mask)
        results = []
        cnt = 0
        for mask in one_hot_mask:
            mask_rect = self.seg2box(mask, cnt)
            results.append(mask_rect)
            cnt += 1
        return results

    def seg2box(self, mask, cnt):
        mask_h, mask_w = mask.shape
        mask_area = mask_h * mask_w
        # print(mask_h, mask_w, np.unique(mask))
        mask = mask.astype(np.uint8)
        ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            area_contour = w * h
            # print("area contours: ", area_contour)
            if area_contour < (0.005*mask_area):
                continue
            xmin, ymin = float(x) / float(mask_w), float(y) / float(mask_h)
            xmax, ymax = float(x+w) / float(mask_w), float(y+h)/ float(mask_h)
            label = get_labelname(self.labelmap, cnt+1)
            rect.append([xmin, ymin, xmax, ymax, label[0]])
        return rect

    def fillhole_flood(self, im_in):
        im_in = im_in.astype(np.uint8)
        im_floodfill = im_in.copy()
        h, w = im_in.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        
        isbreak = False
        for i in range(im_floodfill.shape[0]):
            for j in range(im_floodfill.shape[1]):
                if(im_floodfill[i][j]==0):
                    seedPoint=(i,j)
                    isbreak = True
                    break
            if(isbreak):
                break
        
        cv2.floodFill(im_floodfill, mask, (seedPoint), 1)
        # cv2.floodFill(im_floodfill, mask, (0,0), 1)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = im_in | im_floodfill_inv
        return im_out

    def fillhole(self, im_in):
        im_in = im_in.astype(np.uint8)
        ret, thresh = cv2.threshold(im_in, 0, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im_out = np.zeros(im_in.shape)
        cv2.drawContours(im_out, contours, 0, (255,0,0), -1);
        return im_out


# ('Final Segmentaiton: miou: ', 0.5197170370984521, ' recall: ', 0.7271262796476123, ' precision: ', 0.56541886101382435)
def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    img_dir = args.image_file
    img_path = '/media/data/seg_dataset/fbox/JPEGImages'
    dst_path = '/home/yaok/software/caffe_ssd/result/box/mask_96/union'
    mask_path = '/media/data/seg_dataset/fbox/SemanticLabels'
    xml_path = '/media/data/seg_dataset/fbox/Annotations'

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    n_cls                  = 8
    voc_palette            = vis.make_palette(n_cls)

    # metrics 
    seg_miou, seg_recall, seg_precision = [], [], []

    with open(img_dir) as f:
        img_list = f.readlines()

    for img_name in img_list:
        # img_name = 'IMG_20180305_120305'
        img_name = img_name.strip() + '.png'
        mask_name = img_name.strip()[0:-4] + '_seg.png'
        gt_mask_name = img_name.strip()[0:-4] + '.bmp'
        xml_name = img_name.strip()[0:-4] + '.xml'

        print(img_name)
        full_name = os.path.join(img_path, img_name)
        gt_mask_path = os.path.join(mask_path, gt_mask_name)
        xml_full_path = os.path.join(xml_path, xml_name)

        result, seg_mask = detection.detect(full_name)
        # mask_rect = detection.seg2box_multiclass(mask)

        mask_h, mask_w = seg_mask.shape
        det_mask = np.zeros((mask_h, mask_w, 3))
        gt_mask = cv2.imread(gt_mask_path, 0)
        gt_mask = cv2.resize(gt_mask, (mask_h, mask_w))

        for item in result:
            xmin = int(round(item[0] * mask_w))
            ymin = int(round(item[1] * mask_h))
            xmax = int(round(item[2] * mask_w))
            ymax = int(round(item[3] * mask_h))
            pt_x = int((xmax + xmin) / 2)
            pt_y = int((ymax + ymin) / 2)
            cls_ind = categories[item[-1]]
            det_mask = cv2.rectangle(det_mask, (xmin, ymin), (xmax, ymax), (cls_ind,cls_ind,cls_ind), -1)
        img = cv2.imread(full_name)
        h, w, _ = img.shape
        det_mask_gray = det_mask[:,:,0].astype(np.uint8)

        seg_mask_onehot = detection.one_hot_transform(seg_mask)
        seg_mask_onehot.transpose(1,2,0)
        det_mask_onehot = detection.one_hot_transform(det_mask_gray)
        det_mask_onehot.transpose(1,2,0)
        assert seg_mask_onehot.shape == det_mask_onehot.shape
        union = np.zeros(seg_mask.shape).astype(np.uint8)
        for i in range(1, n_cls):
            union_tmp = cv2.bitwise_or(det_mask_onehot[i-1], seg_mask_onehot[i-1])
            union[np.where(union_tmp == 1)] = i

        miou_seg, rec_seg, prec_seg = seg_scores(gt_mask, union, n_cls)
        seg_miou.append(miou_seg)
        seg_recall.append(rec_seg)
        seg_precision.append(prec_seg)
        print("segmentation metrics: miou: ", miou_seg, " recall: ", \
         rec_seg, " precision: ", prec_seg)

        resize_mask = cv2.resize(union, (h, w), interpolation=cv2.INTER_NEAREST)
        out_im = vis.vis_seg(img, resize_mask, voc_palette)
        dst_mask_name = os.path.join(dst_path, mask_name)
        cv2.imwrite(dst_mask_name, out_im)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Final Segmentaiton: miou: ", np.array(seg_miou).sum()/len(seg_miou), 
                           " recall: ", np.array(seg_recall).sum()/len(seg_recall),
                           " precision: ", np.array(seg_precision).sum()/len(seg_precision))
    # print("Final deteciton: miou: ", np.array(det_miou).sum()/len(det_miou), 
    #                     " recall: ", np.array(det_recall).sum()/len(det_recall),
    #                     " precision: ", np.array(det_precision).sum()/len(det_precision))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',

                            default='/media/data/seg_dataset/fbox/ImageSets/labelmap.prototxt')
    parser.add_argument('--model_def',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/fbox/SSD_384x384_mask96_nonoiseaug/deploy.prototxt')
    parser.add_argument('--image_resize', default=384, type=int)
    parser.add_argument('--model_weights',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/fbox/SSD_384x384_mask96_nonoiseaug/'
                        'VGG_fbox_SSD_384x384_iter_34000.caffemodel') # 6000 34000
    parser.add_argument('--image_file', default='/media/data/seg_dataset/fbox/ImageSets/pics.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
