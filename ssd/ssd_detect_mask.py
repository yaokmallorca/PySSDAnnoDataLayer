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
categories = ['_background',
                'label', 'blue paper tape', 'pink paper tape', 
                'black paper tape', 'lock', 'occupy_filter', 
                'empty_filter']


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
    def one_hot_transform(self, in_masks, ignore=False, ):
        n_cl = 8
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


def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.image_resize, args.labelmap_file)
    img_dir = args.image_file
    img_path = '/media/data/seg_dataset/fbox/JPEGImages'
    dst_path = '/home/yaok/software/caffe_ssd/result/box/mask_96'
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
    det_miou, det_recall, det_precision = [], [], []

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

        result, mask = detection.detect(full_name)
        mask_rect = detection.seg2box_multiclass(mask)

        mask_h, mask_w = mask.shape
        img = cv2.imread(full_name)
        h, w, _ = img.shape
        gt_mask = cv2.imread(gt_mask_path, 0)
        gt_mask = cv2.resize(gt_mask, (mask_h, mask_w))

        # img_mask = img.copy()
        mask_name = os.path.join(dst_path, mask_name)
        # cv2.resize(mask.astype(np.float32), (h, w), interpolation=cv2.INTER_CUBIC).astype(np.int32)
        # img_mask = cv2.resize(img, (mask_h, mask_w), interpolation=cv2.INTER_CUBIC)
        # out_im = vis.vis_seg(img_mask, mask, voc_palette)
        # out_im = cv2.resize(out_im, (h, w), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(mask_name, out_im)
        # out_im.save(mask_name)
        # img = cv2.resize(out_im, (h, w), interpolation = cv2.INTER_CUBIC)
        resize_mask = mask.copy()
        img_mask = img.copy()
        print("seg result: ", gt_mask.shape, mask.shape)
        miou_seg, rec_seg, prec_seg = seg_scores(gt_mask, mask, n_cls)
        seg_miou.append(miou_seg)
        seg_recall.append(rec_seg)
        seg_precision.append(prec_seg)
        print("segmentation metrics: miou: ", miou_seg, " recall: ", \
         rec_seg, " precision: ", prec_seg)
        resize_mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)
        out_im = vis.vis_seg(img_mask, resize_mask, voc_palette)
        cv2.imwrite(mask_name, out_im)

        for item in result:
            xmin = int(round(item[0] * w))
            ymin = int(round(item[1] * h))
            xmax = int(round(item[2] * w))
            ymax = int(round(item[3] * h))
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            pt_x = int((xmax + xmin) / 2)
            pt_y = int((ymax + ymin) / 2)
            cv2.putText(img,item[-1] + str(item[-2]), 
                (pt_x, pt_y), 
                font, 
                fontScale,
                fontColor,
                lineType)
            # print("bbox: ", xmin, ymin, xmax, ymax, str(item[-2]), str(item[-1]))
            dst_name = os.path.join(dst_path, img_name)
        result = np.array(result)

        if len(result) == 0:
            det_miou.append(0)
            det_recall.append(0)
            det_precision.append(0)
            print("detection metrics: miou: ", 0, " recall: ", \
             0, " precision: ", 0)
        else:
            miou_per_img, rec_per_img, prec_per_img = voc_eval_per_img(result, xml_full_path, h, w)
            print("detection metrics: miou: ", miou_per_img, " recall: ", \
             rec_per_img, " precision: ", prec_per_img)
            det_miou.append(miou_per_img)
            det_recall.append(rec_per_img)
            det_precision.append(prec_per_img)
        # print("deteciton metrics: miou: ", np.array(miou_per_img).sum()/(len(miou_per_img)),
        #       " recall: ", np.array(rec_per_img).sum()/(len(rec_per_img)), 
        #       " precision: ", np.array(prec_per_img).sum()/(len(prec_per_img)))
        # for i in range(len(mask_rect)): # (mask_rect.shape[0]):
        #     for box in mask_rect[i]:
        #         xmin = int(round(box[0] * w))
        #         ymin = int(round(box[1] * h))
        #         xmax = int(round(box[2] * w))
        #         ymax = int(round(box[3] * h))
        #         label = box[-1]
        #         # print("seg label: ", label)
        #         box_w = xmax - xmin
        #         box_h = ymax - ymin
        #         area = box_w * box_h
        #         img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        #         pt_x = int((xmax + xmin) / 2)
        #         pt_y = int((ymax + ymin) / 2)
        #         cv2.putText(img, label, (pt_x, pt_y), font, fontScale, fontColor, lineType)
        #         """
        #         result_tmp = result[np.where(result[:, 4][0].astype(np.uint8))]
        #         if len(result_tmp) > 0:
        #             rect = np.array([xmin, ymin, xmax, ymax])
        #             ixmin = np.maximum(result_tmp[:, 0].astype(np.float32), rect[0])
        #             iymin = np.maximum(result_tmp[:, 1].astype(np.float32), rect[1])
        #             ixmax = np.minimum(result_tmp[:, 2].astype(np.float32), rect[2])
        #             iymax = np.minimum(result_tmp[:, 3].astype(np.float32), rect[3])
        #             iw = np.maximum(ixmax - ixmin + 1., 0.)
        #             ih = np.maximum(iymax - iymin + 1., 0.)
        #             inters = iw * ih
        #             uni = ((rect[2] - rect[0] + 1.) * (rect[3] - rect[1] + 1.) +
        #                    (result_tmp[:, 2].astype(np.float32) - result_tmp[:, 0].astype(np.float32) + 1.) *
        #                    (result_tmp[:, 3].astype(np.float32) - result_tmp[:, 1].astype(np.float32) + 1.) - inters)
        #             overlaps = inters / uni
        #             ovmax = np.max(overlaps)
        #             if ovmax < 0.5 and area > 300:
        #                 img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        #         else:
        #             if area > 300:
        #                 img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        #         """
        cv2.imwrite(dst_name, img)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("Final Segmentaiton: miou: ", np.array(seg_miou).sum()/len(seg_miou), 
                           " recall: ", np.array(seg_recall).sum()/len(seg_recall),
                           " precision: ", np.array(seg_precision).sum()/len(seg_precision))
    print("Final deteciton: miou: ", np.array(det_miou).sum()/len(det_miou), 
                        " recall: ", np.array(det_recall).sum()/len(det_recall),
                        " precision: ", np.array(det_precision).sum()/len(det_precision))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_file',

                            default='/media/data/seg_dataset/fbox/ImageSets/labelmap.prototxt')
    parser.add_argument('--model_def',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/fbox/SSD_384x384_96/deploy.prototxt')
    parser.add_argument('--image_resize', default=384, type=int)
    parser.add_argument('--model_weights',
                        default='/home/yaok/software/caffe_ssd/models/VGGNet/fbox/SSD_384x384_96/'
                        'VGG_fbox_SSD_384x384_iter_30000.caffemodel') # 6000 34000
    parser.add_argument('--image_file', default='/media/data/seg_dataset/fbox/ImageSets/val.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
