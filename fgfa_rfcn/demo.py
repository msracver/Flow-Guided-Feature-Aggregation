# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import time

import logging
import pprint
import cv2
from config.config import config as cfg
from config.config import update_config
from utils.image import resize, transform
import numpy as np
from collections import deque


# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fgfa_rfcn/cfgs/fgfa_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet/', cfg.MXNET_VERSION))
import mxnet as mx
import time
from core.tester import im_detect, Predictor, get_resnet_output, prepare_data, draw_all_detection
from symbols import *
from nms.seq_nms import seq_nms
from utils.load_model import load_param
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Flow-Guided Feature Aggregation demo')
    args = parser.parse_args()
    return args

args = parse_args()




def process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, center_image, scales):
    for delta, (scores, boxes, data_dict) in enumerate(pred_result):
        for j in range(1,num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][idx+delta]=cls_dets
            else:
                cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]

        if cfg.TEST.SEQ_NMS==False and  max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][idx + delta][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][idx + delta][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx + delta] = all_boxes[j][idx + delta][keep, :]

            boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, num_classes)]

            out_im = draw_all_detection(center_image, boxes_this_image, classes, scales[delta], cfg)

            return out_im
    return 0


def save_image(output_dir, count, out_im):
    filename = str(count) + '.JPEG'
    cv2.imwrite(output_dir + filename, out_im)

def main():
    # get symbol
    pprint.pprint(cfg)
    cfg.symbol = 'resnet_v1_101_flownet_rfcn'
    model = '/../model/rfcn_fgfa_flownet_vid'
    all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
    max_per_image = cfg.TEST.max_per_image
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)

    # set up class names
    num_classes = 31
    classes = ['__background__','airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    # load demo data

    image_names = glob.glob(cur_path + '/../demo/ILSVRC2015_val_00007010/*.JPEG')
    output_dir = cur_path + '/../demo/rfcn_fgfa/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    for im_name in image_names:
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = cfg.SCALES[0][0]
        max_size = cfg.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
        data.append({'data': im_tensor, 'im_info': im_info,  'data_cache': im_tensor,    'feat_cache': im_tensor})



    # get predictor

    print 'get-predictor'
    data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
    label_names = []

    t1 = time.time()
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_cache', (19, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('feat_cache', ((19, cfg.network.FGFA_FEAT_DIM,
                                                np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int))))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for _ in xrange(len(data))]

    arg_params, aux_params = load_param(cur_path + model, 0, process=True)

    feat_predictors = Predictor(feat_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    aggr_predictors = Predictor(aggr_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = py_nms_wrapper(cfg.TEST.NMS)


    # First frame of the video
    idx = 0
    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    all_boxes = [[[] for _ in range(len(data))]
                 for _ in range(num_classes)]
    data_list = deque(maxlen=all_frame_interval)
    feat_list = deque(maxlen=all_frame_interval)
    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
    # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
    while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
        data_list.append(image)
        feat_list.append(feat)

    vis = False
    file_idx = 0
    thresh = 1e-3
    for idx, element in enumerate(data):

        data_batch = mx.io.DataBatch(data=[element], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, element)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        if(idx != len(data)-1):

            if len(data_list) < all_frame_interval - 1:
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

            else:
                #################################################
                # main part of the loop
                #################################################
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

                prepare_data(data_list, feat_list, data_batch)
                pred_result = im_detect(aggr_predictors, data_batch, data_names, scales, cfg)

                data_batch.data[0][-2] = None
                data_batch.provide_data[0][-2] = ('data_cache', None)
                data_batch.data[0][-1] = None
                data_batch.provide_data[0][-1] = ('feat_cache', None)

                out_im = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, file_idx, max_per_image, vis,
                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                total_time = time.time()-t1
                if (cfg.TEST.SEQ_NMS==False):
                    save_image(output_dir, file_idx, out_im)
                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time /(file_idx+1))
                file_idx += 1
        else:
            #################################################
            # end part of a video                           #
            #################################################

            end_counter = 0
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                data_list.append(image)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                pred_result = im_detect(aggr_predictors, data_batch, data_names, scales, cfg)

                out_im = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, file_idx, max_per_image, vis,
                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)

                total_time = time.time() - t1
                if (cfg.TEST.SEQ_NMS == False):
                    save_image(output_dir, file_idx, out_im)
                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time / (file_idx+1))
                file_idx += 1
                end_counter+=1

    if(cfg.TEST.SEQ_NMS):
        video = [all_boxes[j][:] for j in range(1, num_classes)]
        dets_all = seq_nms(video)
        for cls_ind, dets_cls in enumerate(dets_all):
            for frame_ind, dets in enumerate(dets_cls):
                keep = nms(dets)
                all_boxes[cls_ind + 1][frame_ind] = dets[keep, :]
        for idx in range(len(data)):
            boxes_this_image = [[]] + [all_boxes[j][idx] for j in range(1, num_classes)]
            out_im = draw_all_detection(data[idx][0].asnumpy(), boxes_this_image, classes, scales[0], cfg)
            save_image(output_dir, idx, out_im)

    print 'done'

if __name__ == '__main__':
    main()
