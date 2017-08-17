# --------------------------------------------------------
# Flow-guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuhao Fu, Xizhou Zhu
# --------------------------------------------------------

"""
given a imagenet vid imdb, compute mAP
"""

import numpy as np
import os
import cPickle
import scipy.io as sio
import copy


def parse_vid_rec(filename, classhash, img_ids, defaultIOUthr=0.5, pixelTolerance=10):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['label'] = classhash[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance))
        obj_dict['thr'] = np.min([thr, defaultIOUthr])
        objects.append(obj_dict)
    return {'bbox' : np.array([x['bbox'] for x in objects]),
             'label': np.array([x['label'] for x in objects]),
             'thr'  : np.array([x['thr'] for x in objects]),
             'img_ids': img_ids}


def vid_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vid_eval_motion(multifiles, detpath, annopath, imageset_file, classname_map, annocache, motion_iou_file, motion_ranges, area_ranges, ovthresh=0.5):
    """
    imagenet vid evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """

    with open(imageset_file, 'r') as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
    img_basenames = [x[0] for x in lines]
    gt_img_ids = [int(x[1]) for x in lines]
    classhash = dict(zip(classname_map, range(0,len(classname_map))))

   # load annotations from cache
    if not os.path.isfile(annocache):
        recs = []
        for ind, image_filename in enumerate(img_basenames):
            recs.append(parse_vid_rec(annopath.format('VID/' + image_filename), classhash, gt_img_ids[ind]))
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(img_basenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = cPickle.load(f)

    # read detections
    splitlines = []
    if (multifiles == False):
        with open(detpath, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
    else:
        for det in detpath:
            with open(det, 'r') as f:
                lines = f.readlines()
            splitlines += [x.strip().split(' ') for x in lines]

    splitlines=np.array(splitlines)
    img_ids = splitlines[:,0].astype(int)
    obj_labels = splitlines[:,1].astype(int)
    obj_confs = splitlines[:,2].astype(float)
    obj_bboxes = splitlines[:,3:].astype(float)

    # sort by img_ids
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]

    num_imgs = max(max(gt_img_ids),max(img_ids)) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    # sort by confidence
    for i in range(0, len(img_ids)):
        if i == len(img_ids)-1 or img_ids[i+1] != id:
            conf = obj_confs[start_i:i+1]
            label = obj_labels[start_i:i+1]
            bbox = obj_bboxes[start_i:i+1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids)-1:
                id = img_ids[i+1]
                start_i = i+1

    ov_all = [None] * num_imgs
    # extract objects in :param classname:
    npos = np.zeros(len(classname_map))
    for index, rec in enumerate(recs):
        id = rec['img_ids']
        gt_labels = rec['label']
        gt_bboxes = rec['bbox']
        num_gt_obj = len(gt_labels)

        # calculate total gt for each class
        for x in gt_labels:
            npos[x] += 1  # class: number

        labels = obj_labels_cell[id]
        bboxes = obj_bboxes_cell[id]

        num_obj = 0 if labels is None else len(labels)
        ov_obj = [None] * num_obj
        for j in range(0, num_obj):
            bb = bboxes[j, :]
            ov_gt = np.zeros(num_gt_obj)
            for k in range(0, num_gt_obj):
                bbgt = gt_bboxes[k, :]
                bi = [np.max((bb[0], bbgt[0])), np.max((bb[1], bbgt[1])), np.min((bb[2], bbgt[2])),
                      np.min((bb[3], bbgt[3]))]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                         (bbgt[2] - bbgt[0] + 1.) * \
                         (bbgt[3] - bbgt[1] + 1.) - iw * ih
                    ov_gt[k] = iw * ih / ua
            ov_obj[j] = ov_gt
        ov_all[id] = ov_obj

    # read motion iou
    motion_iou = sio.loadmat(motion_iou_file)
    motion_iou = np.array([[motion_iou['motion_iou'][i][0][j][0] if len(motion_iou['motion_iou'][i][0][j]) != 0 else 0 \
                            for j in range(len(motion_iou['motion_iou'][i][0]))] \
                                for i in range(len(motion_iou['motion_iou']))])

    ap = np.zeros((len(motion_ranges), len(area_ranges), len(classname_map) - 1))
    gt_precent = np.zeros((len(motion_ranges), len(area_ranges), len(classname_map)+1))

    npos_bak = copy.deepcopy(npos)

    for motion_range_id, motion_range in enumerate(motion_ranges):
        for area_range_id, area_range in enumerate(area_ranges):
            tp_cell = [None] * num_imgs
            fp_cell = [None] * num_imgs
            print '==========================================='
            print 'eval_vid_detection :: accumulating: motion [{0:.1f} {1:.1f}], area [{2} {3} {4} {5}]'.format(
                motion_range[0], motion_range[1], np.sqrt(area_range[0]), np.sqrt(area_range[0]), np.sqrt(area_range[1]), np.sqrt(area_range[1]))

            all_motion_iou = np.concatenate(motion_iou, axis=0)
            empty_weight = sum([(all_motion_iou[i] >= motion_range[0]) & (all_motion_iou[i] <= motion_range[1]) for i in range(len(all_motion_iou))]) / float(len(all_motion_iou))

            for index, rec in enumerate(recs):
                id = rec['img_ids']
                gt_labels = rec['label']
                gt_bboxes = rec['bbox']
                gt_thr = rec['thr']
                num_gt_obj = len(gt_labels)
                gt_detected = np.zeros(num_gt_obj)

                gt_motion_iou = motion_iou[index]
                ig_gt_motion = [(gt_motion_iou[i] < motion_range[0]) | (gt_motion_iou[i] > motion_range[1]) for i in range(len(gt_motion_iou))]
                gt_area = [(x[3] - x[1] + 1) * (x[2] - x[0] + 1) for x in gt_bboxes]
                ig_gt_area = [(area < area_range[0]) | (area > area_range[1]) for area in gt_area]

                labels = obj_labels_cell[id]
                bboxes = obj_bboxes_cell[id]

                num_obj = 0 if labels is None else len(labels)
                tp = np.zeros(num_obj)
                fp = np.zeros(num_obj)

                for j in range(0, num_obj):
                    bb = bboxes[j, :]
                    ovmax = -1
                    kmax = -1
                    ovmax_ig = -1
                    ovmax_nig = -1
                    for k in range(0, num_gt_obj):
                        ov = ov_all[id][j][k]
                        if (ov >= gt_thr[k]) & (ov > ovmax) & (not gt_detected[k]) & (labels[j] == gt_labels[k]):
                            ovmax = ov
                            kmax = k
                        if ig_gt_motion[k] & (ov > ovmax_ig):
                            ovmax_ig = ov
                        if (not ig_gt_motion[k]) & (ov > ovmax_nig):
                            ovmax_nig = ov

                    if kmax >= 0:
                        gt_detected[kmax] = 1
                        if (not ig_gt_motion[kmax]) & (not ig_gt_area[kmax]):
                            tp[j] = 1.0
                    else:
                        bb_area = (bb[3] - bb[1] + 1) * (bb[2] - bb[0] + 1)
                        if (bb_area < area_range[0]) | (bb_area > area_range[1]):
                            fp[j] = 0
                            continue

                        if ovmax_nig > ovmax_ig:
                            fp[j] = 1
                        elif ovmax_ig > ovmax_nig:
                            fp[j] = 0
                        elif num_gt_obj == 0:
                            fp[j] = empty_weight
                        else:
                            fp[j] = sum([1 if ig_gt_motion[i] else 0 for i in range(len(ig_gt_motion))]) / float(num_gt_obj)

                tp_cell[id] = tp
                fp_cell[id] = fp

                for k in range(0, num_gt_obj):
                    label = gt_labels[k]
                    if (ig_gt_motion[k]) | (ig_gt_area[k]):
                        npos[label] = npos[label] - 1

            ap[motion_range_id][area_range_id] = calculate_ap(tp_cell, fp_cell, gt_img_ids, obj_labels_cell, obj_confs_cell, classname_map, npos)
            gt_precent[motion_range_id][area_range_id][len(classname_map)] = sum([float(npos[i]) for i in range(len(npos))]) / sum([float(npos_bak[i]) for i in range(len(npos_bak))])
            npos = copy.deepcopy(npos_bak)

    return ap


def boxoverlap(bb, bbgt):
    ov = 0
    iw = np.min((bb[2],bbgt[2])) - np.max((bb[0],bbgt[0])) + 1
    ih = np.min((bb[3],bbgt[3])) - np.max((bb[1],bbgt[1])) + 1
    if iw>0 and ih>0:
        # compute overlap as area of intersection / area of union
        intersect = iw * ih
        ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
               (bbgt[2] - bbgt[0] + 1.) * \
               (bbgt[3] - bbgt[1] + 1.) - intersect
        ov = intersect / ua
    return ov

def calculate_ap(tp_cell, fp_cell, gt_img_ids, obj_labels_cell, obj_confs_cell, classname_map, npos):
    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    cur_ap = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == c])
        tp = np.cumsum(tp_all[obj_labels == c])
        if npos[c] <= 0:
            cur_ap[c] = -1
        else:
            # avoid division by zero in case first detection matches a difficult ground truth
            rec = tp / npos[c]
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            cur_ap[c] = vid_ap(rec, prec)
    return cur_ap[1:]
