# Copyright (c) OpenMMLab. All rights reserved.
import torch


def bbox3d_mapping_back(bboxes, scale_factor, flip_horizontal, flip_vertical):
    """Map bboxes from testing scale to original image scale.

    Args:
        bboxes (:obj:`BaseInstance3DBoxes`): Boxes to be mapped back.
        scale_factor (float): Scale factor.
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.

    Returns:
        :obj:`BaseInstance3DBoxes`: Boxes mapped back.
    """
    new_bboxes = bboxes.clone()
    if flip_horizontal:
        new_bboxes.flip('horizontal')
    if flip_vertical:
        new_bboxes.flip('vertical')
    new_bboxes.scale(1 / scale_factor)

    return new_bboxes


def bbox3d2roi(bbox_list):
    """Convert a list of bounding boxes to roi format.

    Args:
        bbox_list (list[torch.Tensor]): A list of bounding boxes
            corresponding to a batch of images.

    Returns:
        torch.Tensor: Region of interests in shape (n, c), where
            the channels are in order of [batch_ind, x, y ...].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = torch.zeros_like(bboxes)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox3d2result_owp(bboxes, scores, labels, bboxes_ap, scores_ap, labels_ap, attrs=None, test='det'):
    #(bboxes, scores, labels, bboxes_block1, scores_block1, labels_block1, bboxes_block2, scores_block2, labels_block2, bboxes_block3, scores_block3, labels_block3, bboxes_block4, scores_block4, labels_block4, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 5).
        labels (torch.Tensor): Labels with shape (N, ).
        scores (torch.Tensor): Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    if test == 'det':
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu()
            # boxes_3d=bboxes_block4.to('cpu'),
            # scores_3d=scores_block4.cpu(),
            # labels_3d=labels_block4.cpu(),
            # bboxes_3d_block1=bboxes_block1.to('cpu'),
            # scores_3d_block1=scores_block1.cpu(),
            # labels_3d_block1=labels_block1.cpu(),
            # bboxes_3d_block2=bboxes_block2.to('cpu'),
            # scores_3d_block2=scores_block2.cpu(),
            # labels_3d_block2=labels_block2.cpu(),
            # bboxes_3d_block3=bboxes_block3.to('cpu'),
            # scores_3d_block3=scores_block3.cpu(),
            # labels_3d_block3=labels_block3.cpu(),
            # bboxes_3d_block4=bboxes_block4.to('cpu'),
            # scores_3d_block4=scores_block4.cpu(),
            # labels_3d_block4=labels_block4.cpu()
                )
    elif test == 'owp':
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            boxes_ap=bboxes_ap.to('cpu'),
            scores_ap=scores_ap.cpu(),
            labels_ap=labels_ap.cpu(),
        )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


def bbox3d2result(bboxes, scores, labels, attrs=None):
    #(bboxes, scores, labels, bboxes_block1, scores_block1, labels_block1, bboxes_block2, scores_block2, labels_block2, bboxes_block3, scores_block3, labels_block3, bboxes_block4, scores_block4, labels_block4, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 5).
        labels (torch.Tensor): Labels with shape (N, ).
        scores (torch.Tensor): Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """

    result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu()
            # boxes_3d=bboxes_block4.to('cpu'),
            # scores_3d=scores_block4.cpu(),
            # labels_3d=labels_block4.cpu(),
            # bboxes_3d_block1=bboxes_block1.to('cpu'),
            # scores_3d_block1=scores_block1.cpu(),
            # labels_3d_block1=labels_block1.cpu(),
            # bboxes_3d_block2=bboxes_block2.to('cpu'),
            # scores_3d_block2=scores_block2.cpu(),
            # labels_3d_block2=labels_block2.cpu(),
            # bboxes_3d_block3=bboxes_block3.to('cpu'),
            # scores_3d_block3=scores_block3.cpu(),
            # labels_3d_block3=labels_block3.cpu(),
            # bboxes_3d_block4=bboxes_block4.to('cpu'),
            # scores_3d_block4=scores_block4.cpu(),
            # labels_3d_block4=labels_block4.cpu()
                )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict
