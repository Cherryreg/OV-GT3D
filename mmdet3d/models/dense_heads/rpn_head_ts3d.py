import numpy as np

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal, diff_iou_rotated_3d
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss, build_backbone
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner


from icecream import ic
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import axis_aligned_bbox_overlaps_3d
import pdb


@HEADS.register_module()
class RPNHead_TS3D(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_prune_threshold,
                 first_assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', mode="diou"),
                 cls_loss=dict(type='FocalLoss'),
                 keep_loss=dict(type='FocalLoss', reduction='mean', use_sigmoid=True),
                 iou_loss=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(RPNHead_TS3D, self).__init__()
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.first_assigner = build_assigner(first_assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.keep_loss = build_loss(keep_loss)
        self.iou_loss = build_loss(iou_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.n_classes = n_classes
        self.owp=True
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)
    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(in_channels[3], 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(in_channels[2], 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(in_channels[1], 1, kernel_size=1, bias=True, dimension=3)
        ])

        self.iou_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)

        self.pruning = ME.MinkowskiPruning()

        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            # if i < len(in_channels) - 1:
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                        f'out_cls_block_{i}',
                        self.make_block(in_channels[i], out_channels))
            self.__setattr__(
                        f'out_reg_block_{i}',
                        self.make_block(in_channels[i], out_channels))

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        nn.init.normal_(self.iou_conv.kernel, std=.01)

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)


        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) and \
                    ('iou_conv' not in n) and ('keep_conv' not in n) \
                    and ('loss' not in n) and ('unet' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)


    # per level
    def _forward_first(self, x, gt_bboxes, gt_labels, img_metas):

        bboxes_level, bboxes_state = [], []
        for idx in range(len(img_metas)):
            bbox = gt_bboxes[idx]
            bbox_state = torch.cat((bbox.gravity_center, bbox.tensor[:, 3:]), dim=1)
            bbox_level = torch.zeros([len(bbox), 1])
            downsample_times = [5, 4, 3]
            for n in range(len(bbox)):
                bbox_volume = bbox_state[n][3] * bbox_state[n][4] * bbox_state[n][5]
                for i in range(len(downsample_times)):
                    if bbox_volume > 27 * (self.voxel_size * 2 ** downsample_times[i]) ** 3:
                        bbox_level[n] = 3 - i
                        break

            bboxes_level.append(bbox_level)
            bbox_state = torch.cat((bbox_level, bbox_state), dim=1)
            bboxes_state.append(bbox_state)

        bbox_preds, cls_preds, points = [], [], []

        keep_gts, keep_preds, prune_masks = [], [], []

        iou_preds = []
        prune_mask = None
        inputs = x
        x = inputs[-1]
        prune_score = None

        for i in range(len(inputs) - 1, -1, -1):

            if i < len(inputs) - 1:
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)
                keep_gt = []
                for permutation in out_cls.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x = x + x_level
                x = self._prune_training(x, prune_training_keep)

            if i > 0:
                keep_scores = self.keep_conv[i - 1](x)
                prune_training_keep = ME.SparseTensor(
                    -keep_scores.features,
                    coordinate_map_key=keep_scores.coordinate_map_key,
                    coordinate_manager=keep_scores.coordinate_manager)
                keep_pred = keep_scores.features
                prune_inference = keep_pred
                keeps = []
                for permutation in x.decomposition_permutations:
                    keeps.append(keep_pred[permutation])
                keep_preds.append(keeps)
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out_cls = self.__getattr__(f'out_cls_block_{i}')(x)
            out_reg = self.__getattr__(f'out_reg_block_{i}')(x)
            cls_pred, iou_pred, point, prune_training = self._forward_first_single(out_cls)
            ######bbox_pred, _, _, _, _ = self._forward_first_single(out_reg)
            reg_preds = []
            reg_final = self.bbox_conv(out_reg).features
            reg_distance = torch.exp(reg_final[:, 3:6])
            reg_angle = reg_final[:, 6:]
            reg_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
            for permutation in out_reg.decomposition_permutations:
                reg_preds.append(reg_pred[permutation])
            ######
            bbox_preds.append(reg_preds)
            cls_preds.append(cls_pred)
            iou_preds.append(iou_pred)
            points.append(point)

        return bbox_preds[::-1], cls_preds[::-1], iou_preds[::-1], points[::-1], keep_preds[::-1], keep_gts[::-1], bboxes_level
    def _forward_first_single(self, x):
        # reg_final = self.bbox_conv(x).features
        # reg_distance = torch.exp(reg_final[:, 3:6])
        # reg_angle = reg_final[:, 6:]
        # bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features
        prune_training = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        iou_scores = self.iou_conv(x)
        iou_pred = iou_scores.features

        bbox_preds, cls_preds, points = [], [], []
        iou_preds = []
        for permutation in x.decomposition_permutations:
            # bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            iou_preds.append(iou_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)
        # return bbox_preds, cls_preds, points, iou_preds, prune_training
        return cls_preds, iou_preds, points, prune_training



    def _prune_inference(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            # coordinates = x.C.float()
            # interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > 0.1
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x

    def _prune_training(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    @torch.no_grad()
    def _get_keep_voxel(self, input, level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (level - 1):
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for permutation in input.decomposition_permutations:
            point = input.coordinates[permutation][:, 1:] * self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:] * self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                             point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = 7
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (
                                up_level_l * l0 * 2 ** (l + 1)) / 2  # + boxes[..., 4] / 2
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (l + 1)) / 2  # level-2
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (
                                up_level_l * l0 * 2 ** (l + 1)) / 2  # boxes[..., 5] / 2
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (l + 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (
                                up_level_l * l0 * 2 ** (l + 1)) / 2  # boxes[..., 6] / 2
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (l + 1)) / 2

                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

            idx = idx + 1

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene

    def _loss_first(self, bbox_preds, cls_preds, iou_preds, points,
              gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level):
         
        bbox_losses, cls_losses, pos_masks = [], [], []
        iou_losses = []
        # keep loss
        keep_losses = 0
        for i in range(len(img_metas)):
            k_loss = 0
            keep_pred = [x[i] for x in keep_preds]
            keep_gt = [x[i] for x in keep_gts]
            for j in range(len(keep_preds)):
                pred = keep_pred[j]
                gt = (keep_gt[j]).long()

                if gt.sum() != 0:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                    k_loss = torch.mean(keep_loss) / 3 + k_loss
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss


        for i in range(len(img_metas)):
            bbox_loss, cls_loss, iou_loss, pos_mask = self._loss_first_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                bboxes_level=bboxes_level[i])
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
            iou_losses.append(iou_loss)
        return dict(
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)),
            keep_loss=0.01 * keep_losses / len(img_metas),
            iou_losses=torch.mean(torch.stack(iou_losses))
            )   

    def _loss_first_single(self,
                     bbox_preds,
                     cls_preds,
                     iou_preds, 
                     points,
                     gt_bboxes,
                     gt_labels,
                     bboxes_level,
                     img_meta):
        #pdb.set_trace()
        assigned_ids = self.first_assigner.assign(points, gt_bboxes, gt_labels, bboxes_level, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        iou_preds = torch.cat(iou_preds)
        points = torch.cat(points)
        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        avg_factor = max(pos_mask.sum(), 1)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)


        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]

            bbox_preds_to_bbox = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)

            bbox_preds_to_bbox_to_corner = self._bbox_to_loss(bbox_preds_to_bbox)
            pos_bbox_targets_to_corner = self._bbox_to_loss(pos_bbox_targets)

            bbox_loss = self.bbox_loss(bbox_preds_to_bbox_to_corner, pos_bbox_targets_to_corner)

            #####iou_loss
            pos_localization_preds = iou_preds[pos_mask]
            if bbox_preds_to_bbox_to_corner.shape[1] != 6:
                iou_preds_to_targets = diff_iou_rotated_3d(bbox_preds_to_bbox.unsqueeze(0), pos_bbox_targets.unsqueeze(0)).squeeze(0)
            else:
                iou_preds_to_targets = axis_aligned_bbox_overlaps_3d(bbox_preds_to_bbox_to_corner,pos_bbox_targets_to_corner)
                iou_preds_to_targets = torch.diag(iou_preds_to_targets)
            pos_localization_targets = torch.where(iou_preds_to_targets > 0.3, iou_preds_to_targets, torch.zeros_like(iou_preds_to_targets)).unsqueeze(1)

            iou_loss = self.iou_loss(pos_localization_preds, pos_localization_targets, avg_factor=pos_mask.sum())



        else:
            bbox_loss = pos_bbox_preds.sum()
            iou_loss = pos_bbox_preds.sum()
        return bbox_loss, cls_loss, iou_loss, pos_mask

    def _get_bboxes_single_train(self, bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, bboxes_level, img_meta):
        # assigned_ids1 = self.third_assigner.assign(locations, gt_bboxes, img_meta)
        assigned_ids = self.first_assigner.assign(locations, gt_bboxes, gt_labels, bboxes_level, img_meta)
        #pdb.set_trace()
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        locations = torch.cat(locations)

        pos_mask = assigned_ids >= 0


        scores = scores[pos_mask]
        bbox_preds = bbox_preds[pos_mask]
        locations = locations[pos_mask]
        assigned_ids = assigned_ids[pos_mask]

        max_scores, _ = scores.max(dim=1)
        boxes = self._bbox_pred_to_bbox(locations, bbox_preds)

        boxes = torch.cat((
            boxes[:, :3],
            boxes[:, 3:6],
            boxes.new_zeros(boxes.shape[0], 1)), dim=1)
        boxes = img_meta['box_type_3d'](boxes,
                                        with_yaw=False,
                                        origin=(.5, .5, .5))
        return boxes, max_scores, assigned_ids


    def _get_bboxes_train(self, bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, bboxes_level, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_train(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                locations=[x[i] for x in locations],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                bboxes_level=bboxes_level[i],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def _nms(self, bboxes, scores, score_thr, iou_thr, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            # ic(score_thr, iou_thr, scores[:, i])
            ids = scores[:, i] > score_thr
            if not ids.any():
                continue
            try:
                pass
                #_, ids = scores[:i].topk(50)
            except:
                pass

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores, iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
    
    def _nms_1class(self, bboxes, scores, score_thr, iou_thr, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        # ic(score_thr, iou_thr, scores[:, i])
        class_scores, class_labels = torch.max(scores, axis=1)
        class_bboxes = bboxes
        if yaw_flag:
            nms_function = nms3d
        else:
            class_bboxes = torch.cat(
                (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                dim=1)
            nms_function = nms3d_normal
        nms_ids = nms_function(class_bboxes, class_scores, iou_thr)
        nms_bboxes.append(class_bboxes[nms_ids])
        nms_scores.append(class_scores[nms_ids])
        nms_labels.append(
            class_labels[nms_ids])

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _nms_det(self, bboxes, scores, score_thr, iou_thr, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            # ic(score_thr, iou_thr, scores[:, i])
            ids = scores[:, i] > score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores, iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        # nms_bboxes = img_meta['box_type_3d'](
        #     nms_bboxes,
        #     box_dim=box_dim,
        #     with_yaw=with_yaw,
        #     origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _forward_first_test(self, x, img_metas):

        inputs = x
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        iou_preds = []
        keep_scores = None
        for i in range(len(inputs) - 1, -1, -1):
            # out = None
            if i < len(inputs) - 1:
                x = self._prune_inference(x, prune_inference)
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    break
            if i > 0:
                keep_scores = self.keep_conv[i - 1](x)
                keep_pred = keep_scores.features
                prune_inference = keep_pred
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out_cls = self.__getattr__(f'out_cls_block_{i}')(x)
            out_reg = self.__getattr__(f'out_reg_block_{i}')(x)
            cls_pred, iou_pred, point, prune_training = self._forward_first_single(out_cls)
            ######bbox_pred, _, _, _, _ = self._forward_first_single(out_reg)
            reg_preds = []
            reg_final = self.bbox_conv(out_reg).features
            reg_distance = torch.exp(reg_final[:, 3:6])
            reg_angle = reg_final[:, 6:]
            reg_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
            for permutation in out_reg.decomposition_permutations:
                reg_preds.append(reg_pred[permutation])
            ######
            bbox_preds.append(reg_preds)
            cls_preds.append(cls_pred)
            iou_preds.append(iou_pred)
            points.append(point)
        return bbox_preds[::-1], cls_preds[::-1], iou_preds[::-1], points[::-1]


    def _nms_insseg(self, bboxes, scores, cfg, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def forward_train(self, batch_dict):
        # first stage
        x = batch_dict['backbone_feat'][1:]
        gt_bboxes = batch_dict['gt_bboxes_3d']
        gt_labels = batch_dict['gt_labels_3d']
        img_metas = batch_dict['img_metas']

        bbox_preds, cls_preds, iou_preds, locations, keep_preds, keep_gts, bboxes_level = self._forward_first(
            x, gt_bboxes, gt_labels, img_metas)
        losses = self._loss_first(bbox_preds, cls_preds, iou_preds, locations,
                                  gt_bboxes, gt_labels, img_metas,
                                  keep_preds, keep_gts, bboxes_level,
                                  )

        result = {}
        bbox_list = self._get_bboxes_det(bbox_preds, cls_preds, iou_preds, locations, img_metas)
        result['pred_bbox_list'] = bbox_list
        bbox_list_train = self._get_bboxes_train(bbox_preds, cls_preds, locations, gt_bboxes, gt_labels, bboxes_level, img_metas)
        result['pred_bbox_list_train'] = bbox_list_train

        return result, losses


    def forward_test(self, batch_dict):
        # first stage
        x = batch_dict['backbone_feat'][1:]
        img_metas = batch_dict['img_metas']
        result = {}
        bbox_preds, cls_preds, iou_preds, pc = self._forward_first_test(x, img_metas)

        # bbox_list = self._get_bboxes_det(bbox_preds, cls_preds, iou_preds, pc, img_metas)
        
        
        result['pred_bbox_list'] = self._get_bboxes_det(bbox_preds, cls_preds, iou_preds, pc, img_metas) 
        result['pred_bbox_list_1st'] = self._get_bboxes_det2(bbox_preds, cls_preds, iou_preds, pc, img_metas) 
        result['pred_bbox_list_2st'] = self._get_bboxes_2st(bbox_preds, cls_preds, iou_preds, pc, img_metas) 
        return result
    
    def forward_test_openvoc(self, batch_dict):
        result = {}
        x = batch_dict['backbone_feat'][1:]
        img_metas = batch_dict['img_metas']
        seg_feats = batch_dict['semantic_feat']
        text_features_file = batch_dict['text_features_file']
        field = batch_dict['field']
        pc_init = batch_dict['points'][0]
        bbox_preds, cls_preds, iou_preds, pc = self._forward_first_test(x, img_metas)
        bbox_list_det = self._get_bboxes_det_openvoc(seg_feats, text_features_file, field, pc_init, bbox_preds, cls_preds, iou_preds, pc, img_metas)
        
        result['pred_bbox_list_1st'] = bbox_list_det
        return result
    
    
    def _get_bboxes_single_det(self, bbox_preds, cls_preds, iou_preds, points, img_meta):
        
        cls_preds = torch.cat(cls_preds).sigmoid()
        iou_preds = torch.cat(iou_preds).sigmoid()
        theta = 0.7
        scores = pow(iou_preds, theta) * pow(cls_preds, (1-theta))

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)        
        if len(scores) > self.test_cfg.test_nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.test_nms_pre)
            bbox_pred_ar = bbox_preds[ids]
            score_ar = scores[ids]
            point_ar = points[ids]
        else:
            bbox_pred_ar = bbox_preds
            score_ar = scores
            point_ar = points
        labels_ar = torch.zeros_like(score_ar)
        boxes_ar = self._bbox_pred_to_bbox(point_ar, bbox_pred_ar)
        boxes_ar, score_ar, labels_ar = self._nms_det(boxes_ar, score_ar, self.test_cfg.test_score_thr, self.test_cfg.test_iou_thr, img_meta)

        boxes_ar = torch.cat((boxes_ar, torch.zeros((boxes_ar.shape[0],1),dtype=boxes_ar.dtype, device=boxes_ar.device)), dim=1)



        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_pred_ap = bbox_preds[ids]
            score_ap = scores[ids]
            point_ap = points[ids]
        boxes_ap = self._bbox_pred_to_bbox(point_ap, bbox_pred_ap)
        boxes_ap, scores_ap, labels_ap = self._nms_det(boxes_ap, score_ap, self.test_cfg.score_thr, self.test_cfg.iou_thr, img_meta)
        if boxes_ap.shape[1] != 7:
            boxes_ap = torch.cat((boxes_ap, torch.zeros((boxes_ap.shape[0],1),dtype=boxes_ap.dtype, device=boxes_ap.device)), dim=1)
        return boxes_ar, score_ar, labels_ar, boxes_ap, scores_ap, labels_ap

    def _get_bboxes_det(self, bbox_preds, cls_preds, iou_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_det(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results
    
    def _get_bboxes_single_det2(self, bbox_preds, cls_preds, iou_preds, points, img_meta):
        cls_preds = torch.cat(cls_preds).sigmoid()
        iou_preds = torch.cat(iou_preds).sigmoid()
        theta = 0.7
        scores = pow(iou_preds, theta) * pow(cls_preds, (1-theta))

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_pred_ap = bbox_preds[ids]
            score_ap = scores[ids]
            point_ap = points[ids]
        boxes_ap = self._bbox_pred_to_bbox(point_ap, bbox_pred_ap)
        boxes_ap, scores_ap, labels_ap = self._nms(boxes_ap, score_ap, self.test_cfg.score_thr, self.test_cfg.iou_thr, img_meta)
        return boxes_ap, scores_ap, labels_ap

    def _get_bboxes_det2(self, bbox_preds, cls_preds, iou_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_det2(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results
    
    
    
    def _get_bboxes_single_2st(self, bbox_preds, cls_preds, iou_preds, points, img_meta):
        cls_preds = torch.cat(cls_preds).sigmoid()
        iou_preds = torch.cat(iou_preds).sigmoid()
        theta = 0.7
        scores = pow(iou_preds, theta) * pow(cls_preds, (1-theta))

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)

        max_scores, _ = scores.max(dim=1)
        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        if boxes.shape[1] != 7:
            boxes = torch.cat((boxes, torch.zeros((boxes.shape[0],1),dtype=boxes.dtype, device=boxes.device)), dim=1)
        return boxes, scores

    def _get_bboxes_2st(self, bbox_preds, cls_preds, iou_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single_2st(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results
    
    def _get_bboxes_single_det_openvoc(self, seg_feats, text_features_file, field, pc_init, bbox_preds, cls_preds, iou_preds, points, img_meta):
        
        text_features = torch.load(text_features_file).to(pc_init.device)
        inverse_mapping = field.inverse_mapping(seg_feats.coordinate_map_key).long()
        predictions = seg_feats.features[inverse_mapping, :]
        # pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

        # scores = torch.cat(cls_preds).sigmoid()
        cls_preds = torch.cat(cls_preds).sigmoid()
        iou_preds = torch.cat(iou_preds).sigmoid()
        #TODO:theta取值
        theta = 0.3
        scores = pow(iou_preds, theta) * pow(cls_preds, (1-theta))

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_pred_ap = bbox_preds[ids]
            score_ap = scores[ids]
            point_ap = points[ids]
        boxes_ap = self._bbox_pred_to_bbox(point_ap, bbox_pred_ap)
        
        n_points = pc_init.shape[0]
        n_boxes = boxes_ap.shape[0]
        n_reg = boxes_ap.shape[1]
        pc_expand =  pc_init.unsqueeze(1).expand(n_points, n_boxes, 3)
        boxes_expand = boxes_ap.expand(n_points, n_boxes, n_reg)
        if boxes_expand.shape[2] == 6:
            pad = torch.zeros((boxes_expand.shape[0],  boxes_expand.shape[1], 1), device=boxes_expand.device)
            boxes_expand = torch.cat((boxes_expand, pad), dim=2)
        face_distances = self._get_face_distances(pc_expand, boxes_expand)
        inside_box_condition = face_distances.min(dim=-1).values > 0
        mask_list = torch.unbind(inside_box_condition, dim=1)
        mask = torch.zeros(n_boxes).bool()
        box_feat = []
        for i in range(len(mask_list)):
            mask_ = mask_list[i]
            if mask_.sum() > 0:
                #point_feat  max pooling-->  box_feat  -->   box_label
                feat_ = predictions[mask_]               
                # pooled_feature, _ = torch.max(feat_, dim=0)    
                pooled_feature = torch.mean(feat_, dim=0)
                pooled_cls = (pooled_feature/(pooled_feature.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                # point_cls  -->  box_cls
                # pre_cls_ = pred_distill[mask_]
                # pooled_cls = torch.mean(pre_cls_, dim=0)
                #### pooled_cls, _ = torch.max(pre_cls_, dim=0)
                box_feat.append(pooled_cls)
                mask[i] = True
        box_feat = torch.stack(box_feat)

        score_ap = score_ap[mask] * box_feat
        boxes_ap, scores_ap, labels_ap = self._nms(boxes_ap[mask], score_ap, self.test_cfg.score_thr, self.test_cfg.test_iou_thr, img_meta)
        
        # del text_features, predictions, pc_expand, box_feat, boxes_expand  # 删除不再需要的变量
        # torch.cuda.empty_cache()  # 释放显存
        
        return boxes_ap, scores_ap, labels_ap
    
    def _get_bboxes_single_det_openvoc_owp(self, seg_feats, text_features_file, field, pc_init, bbox_preds, cls_preds, iou_preds, points, img_meta):
        if self.test_cfg.train_stage != '1st':
            text_features = torch.load(text_features_file).to(pc_init.device)
            inverse_mapping = field.inverse_mapping(seg_feats.coordinate_map_key).long()
            predictions = seg_feats.features[inverse_mapping, :]
        # pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

        # scores = torch.cat(cls_preds).sigmoid()
        cls_preds = torch.cat(cls_preds).sigmoid()
        iou_preds = torch.cat(iou_preds).sigmoid()
        #TODO:theta取值
        theta = 0.7
        scores = pow(iou_preds, theta) * pow(cls_preds, (1-theta))

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)
        #pdb.set_trace()
        ##################AR
        if len(scores) > self.test_cfg.test_nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.test_nms_pre)
            bbox_pred_ar = bbox_preds[ids]
            score_ar = scores[ids]
            point_ar = points[ids]
        else:
            point_ar = points
            bbox_pred_ar = bbox_preds
            score_ar = scores
        boxes_ar = self._bbox_pred_to_bbox(point_ar, bbox_pred_ar)
        boxes_ar, score_ar, labels_ar = self._nms(boxes_ar, score_ar, self.test_cfg.score_thr, 0.5, img_meta)
        
        if img_meta['sample_idx'] == 'scene0568_00' and False:
            
            import numpy as np
            pad = torch.zeros((boxes_ar.tensor.shape[0], 1), dtype=torch.float32, device=boxes_ar.device)
            boxes_ar_new = torch.cat((boxes_ar.tensor, pad), dim=1)

            np.save("ts3d_boxes_ar_nms.npy", boxes_ar_new.cpu().numpy())
            _, ids = score_ar.topk(50)
            boxes_top10 = boxes_ar[ids]
            scores_top10 = score_ar[ids]
            np.save("ts3d_boxes_ar_nms_top50.npy", boxes_top10.tensor.cpu().numpy())

            print(boxes_top10)
            print(scores_top10)
            pdb.set_trace()

        
        labels_ar = torch.zeros_like(score_ar)
        #boxes_ar = img_meta['box_type_3d'](
        #    boxes_ar,
        #    box_dim=6,
        #    with_yaw=False,
        #    origin=(.5, .5, .5))
        #pdb.set_trace()
        #boxes_ar, score_ar, labels_ar = self._nms(boxes_ar, score_ar, self.test_cfg.score_thr, 0.5, img_meta)

        #######################AR END
        if len(scores) > 300 > 0:
            _, ids = max_scores.topk(300)
            bbox_pred_ap = bbox_preds[ids]
            score_ap = scores[ids]
            point_ap = points[ids]
        boxes_ap = self._bbox_pred_to_bbox(point_ap, bbox_pred_ap)
        '''
        n_points = pc_init.shape[0]
        n_boxes = boxes_ap.shape[0]
        pc_expand =  pc_init.unsqueeze(1).expand(n_points, n_boxes, 3)
        boxes_expand = boxes_ap.expand(n_points, n_boxes, 6)
        face_distances = self._get_face_distances(pc_expand, boxes_expand)
        inside_box_condition = face_distances.min(dim=-1).values > 0
        mask_list = torch.unbind(inside_box_condition, dim=1)
        mask = torch.zeros(n_boxes).bool()
        box_feat = []
        for i in range(len(mask_list)):
            mask_ = mask_list[i]
            if mask_.sum() > 0:
                #point_feat  max pooling-->  box_feat  -->   box_label
                feat_ = predictions[mask_]               
                # pooled_feature, _ = torch.max(feat_, dim=0)    
                pooled_feature = torch.mean(feat_, dim=0)
                pooled_cls = (pooled_feature/(pooled_feature.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                # point_cls  -->  box_cls
                # pre_cls_ = pred_distill[mask_]
                # pooled_cls = torch.mean(pre_cls_, dim=0)
                #### pooled_cls, _ = torch.max(pre_cls_, dim=0)
                box_feat.append(pooled_cls)
                mask[i] = True
        box_feat = torch.stack(box_feat)
        '''
        #score_ap = score_ap[mask] * box_feat
        score_ap = score_ap
        
        #boxes_ap, scores_ap, labels_ap = self._nms(boxes_ap[mask], score_ap, self.test_cfg.score_thr, 0.5, img_meta)
        boxes_ap, scores_ap, labels_ap = self._nms(boxes_ap, score_ap, self.test_cfg.score_thr, 0.5, img_meta)
        return boxes_ar, score_ar, labels_ar, boxes_ap, scores_ap, labels_ap
    
    def _get_bboxes_det_openvoc(self, seg_feats, text_features_file, field, pc_init, bbox_preds, cls_preds, iou_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            if not self.test_cfg.owp:
                result = self._get_bboxes_single_det_openvoc(
                    seg_feats, text_features_file, field, pc_init,
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    iou_preds=[x[i] for x in iou_preds],
                    points=[x[i] for x in points],
                    img_meta=img_metas[i])
                results.append(result)
            else:
                result = self._get_bboxes_single_det_openvoc_owp(
                    seg_feats, text_features_file, field, pc_init,
                    bbox_preds=[x[i] for x in bbox_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    iou_preds=[x[i] for x in iou_preds],
                    points=[x[i] for x in points],
                    img_meta=img_metas[i])
                results.append(result)
        return results
    
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        shift = torch.stack(
            (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
             points[..., 2] - boxes[..., 2]),
            dim=-1).permute(1, 0, 2)
        shift = rotation_3d_in_axis(
            shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)
    

@BBOX_ASSIGNERS.register_module()
class TS3DInstanceAssigner_DetInsseg:
    def __init__(self, top_pts_threshold):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        # self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, bboxes_level, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        # label2level = gt_labels.new_tensor(self.label2level)
        # label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        # point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        bboxes_level = bboxes_level.squeeze(1)
        label_levels = bboxes_level.unsqueeze(0).expand(n_points, n_boxes).to(points.device)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: tonly closest object to poin
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds