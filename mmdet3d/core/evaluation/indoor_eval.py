# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
from icecream import ic
import pdb

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, iou_thr=None):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret


def eval_map_recall(pred, gt, ovthresh=None):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """

    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)

    return recall, precision, ap

def eval_map_recall_owp(pred, gt, ovthresh=None):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """
    ret_values = {}
    for classname in [0]:
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for label in [0]:
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)

    return recall, precision, ap



def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_type_3d=None,
                box_mode_3d=None,
                owp=False):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - boxes_3d (:obj:`BaseInstance3DBoxes`):
                3D bounding boxes in Depth coordinate.
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (dict): Map from label to category.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Return:
        dict[str, float]: Dict of results.
    """
    assert len(dt_annos) == len(gt_annos)
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    pred_small = {}  # map {class_id: pred}
    gt_small = {}  # map {class_id: gt}
    pred_other = {}  # map {class_id: pred}
    gt_other = {}  # map {class_id: gt}
    pred_ap = {}
    pred_small_ap = {}
    pred_other_ap = {}
    
    owp = True
    if not owp:
        for img_id in range(len(dt_annos)):
            # parse detected annotations
            det_anno = dt_annos[img_id]
            for i in range(len(det_anno['labels_3d'])):
                label = det_anno['labels_3d'].numpy()[i]
                bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
                score = det_anno['scores_3d'].numpy()[i]
                if label not in pred:
                    pred[int(label)] = {}
                if img_id not in pred[label]:
                    pred[int(label)][img_id] = []
                if label not in gt:
                    gt[int(label)] = {}
                if img_id not in gt[label]:
                    gt[int(label)][img_id] = []
                pred[int(label)][img_id].append((bbox, score))
            gt_anno = gt_annos[img_id]
            if gt_anno['gt_num'] != 0:
                gt_boxes = box_type_3d(
                    gt_anno['gt_boxes_upright_depth'],
                    box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
                labels_3d = gt_anno['class']
            else:
                gt_boxes = box_type_3d(np.array([], dtype=np.float32))
                labels_3d = np.array([], dtype=np.int64)

            for i in range(len(labels_3d)):
                label = labels_3d[i]
                bbox = gt_boxes[i]
                if label not in gt:
                    gt[label] = {}
                if img_id not in gt[label]:
                    gt[label][img_id] = []
                gt[label][img_id].append(bbox)
            del bbox

        rec, prec, ap = eval_map_recall(pred, gt, metric)
        ret_dict = dict()
        header = ['classes']
        table_columns = [[label2cat[label]
                          for label in ap[0].keys()] + ['Overall']]
        for i, iou_thresh in enumerate(metric):
            header.append(f'AP_{iou_thresh:.2f}')
            header.append(f'AR_{iou_thresh:.2f}')
            rec_list = []
            for label in ap[i].keys():
                ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap[i][label][0])
            ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap[i].values())))

            table_columns.append(list(map(float, list(ap[i].values()))))
            table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

            for label in rec[i].keys():
                ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                    rec[i][label][-1])
                rec_list.append(rec[i][label][-1])
            ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.mean(rec_list))

            table_columns.append(list(map(float, rec_list)))
            table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        return ret_dict
    else:
        assert len(dt_annos) == len(gt_annos)
        pred = {}  # map {class_id: pred}
        gt = {}  # map {class_id: gt}
        pred_ap = {}
        for img_id in range(len(dt_annos)):
            # parse gt annotations
            gt_anno = gt_annos[img_id]
            if gt_anno['gt_num'] != 0:
                gt_boxes = box_type_3d(
                    gt_anno['gt_boxes_upright_depth'],
                    box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
                labels_3d = gt_anno['class']
            else:
                gt_boxes = box_type_3d(np.array([], dtype=np.float32))
                labels_3d = np.array([], dtype=np.int64)
            for i in range(len(labels_3d)):
                label = labels_3d[i]
                bbox = gt_boxes[i]
                if label not in gt:
                    gt[label] = {}
                if img_id not in gt[label]:
                    gt[label][img_id] = []
                gt[label][img_id].append(bbox)
        
        #proposal_nums = (50, 100, 300, 500, 1000)
        proposal_nums = (10, 30, 100, 200, 300)
        header = ['classes']
        table_columns = [['object', 'Overall']]
        ret_dict = dict()
        ###############AP
        for img_id in range(len(dt_annos)):
            # parse detected annotations
            det_anno = dt_annos[img_id]
            for i in range(len(det_anno['labels_ap'])):
                label = det_anno['labels_ap'].numpy()[i]
                bbox = det_anno['boxes_ap'].convert_to(box_mode_3d)[i]
                score = det_anno['scores_ap'].numpy()[i]
                if label not in pred_ap:
                    pred_ap[int(label)] = {}
                if img_id not in pred_ap[label]:
                    pred_ap[int(label)][img_id] = []
                if label not in gt:
                    gt[int(label)] = {}
                if img_id not in gt[label]:
                    gt[int(label)][img_id] = []
                pred_ap[int(label)][img_id].append((bbox, score))
        rec, prec, ap = eval_map_recall(pred_ap, gt, metric)
        #import pdb
        #pdb.set_trace()
        for i, iou_thresh in enumerate(metric):
            header.append(f'AP@{iou_thresh:.2f}')
            for label in ap[i].keys():
                ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap[i][label][0])
            ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap[i].values())))

            table_columns.append(list(map(float, list(ap[i].values()))))
            table_columns[-1] = [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]
        #############################################
        for proposals in proposal_nums:
            for img_id in range(len(dt_annos)):
                # parse detected annotations
                det_anno = dt_annos[img_id]
                prop_num = min(proposals, len(det_anno['labels_3d']))
                #print(proposals, len(det_anno['labels_3d']))
                for i in range(prop_num):
                    try:
                        label = det_anno['labels_3d'].numpy()[i][0]
                        #print(label, det_anno['labels_ap'].numpy()[i], end=' ')
                        bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
                        score = det_anno['scores_3d'].numpy()[i][0]
                    except:
                        label = det_anno['labels_3d'].numpy()[i]
                        #print(label, det_anno['labels_ap'].numpy()[i], end=' ')
                        bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
                        score = det_anno['scores_3d'].numpy()[i]

                    #print(score)
                    #print(bbox.tensor)
                    if label not in pred:
                        pred[int(label)] = {}
                    if img_id not in pred[label]:
                        pred[int(label)][img_id] = []
                    if label not in gt:
                        gt[int(label)] = {}
                    if img_id not in gt[label]:
                        gt[int(label)][img_id] = []
                    pred[int(label)][img_id].append((bbox, score))
                #pdb.set_trace()
            rec, prec, ap = eval_map_recall(pred, gt, metric)
            header.append(f'AR_{proposals:.0f}')
            rec_list = []
            for label in rec[1].keys():
                ret_dict[f'{label2cat[label]}_rec_{proposals:.0f}'] = float(
                    rec[1][label][-1])
                rec_list.append(rec[1][label][-1])
            #pdb.set_trace()
            ret_dict[f'AR_{proposals:.0f}'] = float(np.mean(rec_list))
            # table_columns.append(list(map(float, rec_list)))
            table_columns.append(list(map(float, rec_list)))
            table_columns[-1] = [ret_dict[f'AR_{proposals:.0f}']]
            table_columns[-1] += [ret_dict[f'AR_{proposals:.0f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        return ret_dict

        '''
        for img_id in range(len(dt_annos)):
            # parse gt annotations
            gt_anno = gt_annos[img_id]
            if gt_anno['gt_num'] != 0:
                gt_boxes = box_type_3d(
                    gt_anno['gt_boxes_upright_depth'],
                    box_dim=gt_anno['gt_boxes_upright_depth'].shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
                labels_3d = gt_anno['class']
            else:
                gt_boxes = box_type_3d(np.array([], dtype=np.float32))
                labels_3d = np.array([], dtype=np.int64)

            for i in range(len(labels_3d)):
                label = labels_3d[i]
                bbox = gt_boxes[i]
                if label not in gt:
                    gt[label] = {}
                    gt_small[label] = {}
                    gt_other[label] = {}
                if img_id not in gt[label]:
                    gt[label][img_id] = []
                    gt_small[label][img_id] = []
                    gt_other[label][img_id] = []

                gt[label][img_id].append(bbox)
                if bbox.volume > 0.05:
                    gt_other[label][img_id].append(bbox)
                else:
                    gt_small[label][img_id].append(bbox)

        proposal_nums = (50, 100, 300, 500, 1000)
        header = ['classes']
        table_columns = [['Small_object', 'Other_object', 'Overall']]
        ret_dict = dict()
        ret_dict_small = dict()
        ret_dict_other = dict()
     ###############AP
        for img_id in range(len(dt_annos)):
            # parse detected annotations
            det_anno = dt_annos[img_id]
            for i in range(len(det_anno['labels_ap'])):
                label = det_anno['labels_ap'].numpy()[i]
                bbox = det_anno['boxes_ap'].convert_to(box_mode_3d)[i]
                score = det_anno['scores_ap'].numpy()[i]
                if label not in pred_ap:
                    pred_ap[int(label)] = {}
                    pred_small_ap[int(label)] = {}
                    pred_other_ap[int(label)] = {}

                if img_id not in pred_ap[label]:
                    pred_ap[int(label)][img_id] = []
                    pred_small_ap[int(label)][img_id] = []
                    pred_other_ap[int(label)][img_id] = []


                if label not in gt:
                    gt[int(label)] = {}
                    gt_small[int(label)] = {}
                    gt_other[int(label)] = {}
                if img_id not in gt[label]:
                    gt[int(label)][img_id] = []
                    gt_small[int(label)][img_id] = []
                    gt_other[int(label)][img_id] = []
                pred_ap[int(label)][img_id].append((bbox, score))
                if bbox.volume > 0.05:
                    pred_other_ap[int(label)][img_id].append((bbox, score))
                else:
                    pred_small_ap[int(label)][img_id].append((bbox, score))

        rec, prec, ap = eval_map_recall(pred_ap, gt, metric)
        rec_small, prec_small, ap_small = eval_map_recall(pred_small_ap, gt_small, metric)
        rec_other, prec_other, ap_other = eval_map_recall(pred_other_ap, gt_other, metric)
        for i, iou_thresh in enumerate(metric):
            header.append(f'AP@{iou_thresh:.2f}')

            for label in ap[i].keys():
                ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap[i][label][0])
                ret_dict_small[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap_small[i][label][0])
                ret_dict_other[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                    ap_other[i][label][0])

            ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap[i].values())))
            ret_dict_small[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap_small[i].values())))
            ret_dict_other[f'mAP_{iou_thresh:.2f}'] = float(
                np.mean(list(ap_other[i].values())))

            table_columns.append(list(map(float, list(ap[i].values()))))
            table_columns[-1] = [ret_dict_small[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] += [ret_dict_other[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]


            # ic(ret_dict, ret_dict_other, ret_dict_small)





#############################################
        # proposal_nums = (50, 100, 300, 500, 1000)
        # header = ['classes']
        # table_columns = [['Small_object', 'Other_object', 'Overall']]
        for proposals in proposal_nums:
            for img_id in range(len(dt_annos)):
                # parse detected annotations
                det_anno = dt_annos[img_id]
                prop_num = min(proposals, len(det_anno['labels_3d']))
                for i in range(prop_num):
                    label = det_anno['labels_3d'].numpy()[i]
                    bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
                    score = det_anno['scores_3d'].numpy()[i]
                    if label not in pred:
                        pred[int(label)] = {}
                        pred_small[int(label)] = {}
                        pred_other[int(label)] = {}

                    if img_id not in pred[label]:
                        pred[int(label)][img_id] = []
                        pred_small[int(label)][img_id] = []
                        pred_other[int(label)][img_id] = []
                    if label not in gt:
                        gt[int(label)] = {}
                        gt_small[int(label)] = {}
                        gt_other[int(label)] = {}
                    if img_id not in gt[label]:
                        gt[int(label)][img_id] = []
                        gt_small[int(label)][img_id] = []
                        gt_other[int(label)][img_id] = []
                    pred[int(label)][img_id].append((bbox, score))
                    if bbox.volume > 0.05:
                        pred_other[int(label)][img_id].append((bbox, score))
                    else:
                        pred_small[int(label)][img_id].append((bbox, score))

            rec, prec, ap = eval_map_recall(pred, gt, metric)
            rec_small, prec_small, ap_small = eval_map_recall(pred_small, gt_small, metric)
            rec_other, prec_other, ap_other = eval_map_recall(pred_other, gt_other, metric)


            # if proposals == 50:
            #     for i, iou_thresh in enumerate(metric):
            #         header.append(f'AP@{iou_thresh:.2f}_{proposals:.0f}')
            #
            #         for label in ap[i].keys():
            #             ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
            #                 ap[i][label][0])
            #             ret_dict_small[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
            #                 ap_small[i][label][0])
            #             ret_dict_other[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
            #                 ap_other[i][label][0])
            #
            #         ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
            #             np.mean(list(ap[i].values())))
            #         ret_dict_small[f'mAP_{iou_thresh:.2f}'] = float(
            #             np.mean(list(ap_small[i].values())))
            #         ret_dict_other[f'mAP_{iou_thresh:.2f}'] = float(
            #             np.mean(list(ap_other[i].values())))
            #
            #
            #         table_columns.append(list(map(float, list(ap[i].values()))))
            #         table_columns[-1] = [ret_dict_small[f'mAP_{iou_thresh:.2f}']]
            #         table_columns[-1] += [ret_dict_other[f'mAP_{iou_thresh:.2f}']]
            #         table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
            #         table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]


            header.append(f'AR_{proposals:.0f}')
            rec_list = []
            rec_list_small = []
            rec_list_other = []
            for label in rec[1].keys():
                ret_dict[f'{label2cat[label]}_rec_{proposals:.0f}'] = float(
                    rec[1][label][-1])
                rec_list.append(rec[1][label][-1])

                ret_dict_small[f'{label2cat[label]}_rec_{proposals:.0f}'] = float(
                    rec_small[1][label][-1])
                rec_list_small.append(rec_small[1][label][-1])

                ret_dict_other[f'{label2cat[label]}_rec_{proposals:.0f}'] = float(
                    rec_other[1][label][-1])
                rec_list_other.append(rec_other[1][label][-1])
            ret_dict[f'AR_{proposals:.0f}'] = float(np.mean(rec_list))
            ret_dict_small[f'AR_{proposals:.0f}'] = float(np.mean(rec_list_small))
            ret_dict_other[f'AR_{proposals:.0f}'] = float(np.mean(rec_list_other))


            # table_columns.append(list(map(float, rec_list)))
            table_columns.append(list(map(float, rec_list)))
            table_columns[-1] = [ret_dict_small[f'AR_{proposals:.0f}']]
            table_columns[-1] += [ret_dict_other[f'AR_{proposals:.0f}']]
            table_columns[-1] += [ret_dict[f'AR_{proposals:.0f}']]
            table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]




    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    
    return ret_dict
    '''
