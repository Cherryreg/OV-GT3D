# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
import numpy as np

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.core import bbox3d2result, bbox3d2result_owp
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector
import torch
from mmcv.ops import nms3d, nms3d_normal
from icecream import ic
import pdb
from data.scannet.scannet_ovd_sv.Visualize_pc_box import _write_obj, write_oriented_bbox
from pynvml import *
torch.set_printoptions(profile="full")
@DETECTORS.register_module()
class MinkTwoStage3DDetectorPC2OBJ_V2_SV(Base3DDetector):
    r"""Single stage detector based on MinkowskiEngine `GSDN
    <https://arxiv.org/abs/2006.12356>`_.

    Args:
        backbone (dict): Config of the backbone.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        neck (dict): Config of the neck.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 backbone,             
                 voxel_size,
                 train_stage='1st',
                 distill_loss_weight=1,
                 rpn_head=None,
                 semantic_head=None,
                 roi_head=None,
                 text_features_file=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(MinkTwoStage3DDetectorPC2OBJ_V2_SV, self).__init__(init_cfg)
        self.train_stage = train_stage
        self.backbone = build_backbone(backbone)

        rpn_head.update(train_cfg=train_cfg)
        rpn_head.update(test_cfg=test_cfg)
        self.rpn_head = build_head(rpn_head)

        semantic_head.update(train_cfg=train_cfg)
        semantic_head.update(test_cfg=test_cfg)
        self.semantic_head = build_head(semantic_head)

        if self.train_stage != '1st':
            roi_head.update(train_cfg=train_cfg)
            roi_head.update(test_cfg=test_cfg)
            self.roi_head = build_head(roi_head)

        self.text_features_file = text_features_file
        self.test_cfg = test_cfg       
        self.voxel_size = voxel_size
        self.distill_loss_weight = distill_loss_weight
        self.init_weights()

    def init_weights(self, pretrained=None):
        self.rpn_head.init_weights()
        self.semantic_head.init_weights()
        
        # if self.train_stage == '1st':
        #     for name, param in self.named_parameters():
        #         if "img_model"  in name:
        #             param.requires_grad=False
        if self.train_stage == '1st':
            for name, param in self.named_parameters():
                if "img_model"  in name:
                    param.requires_grad=False
        elif self.train_stage == '2st':
            self.roi_head.init_weights()
            for name, param in self.named_parameters():
                if "clip_header" not in name:
                    param.requires_grad=False
            self.backbone.eval()
            self.rpn_head.eval()
            self.semantic_head.eval()
            self.roi_head.img_model.eval()
        elif self.train_stage == '3st':
            self.roi_head.init_weights()
            for name, param in self.named_parameters():
                if "3st" not in name:
                    param.requires_grad=False
            self.backbone.eval()
            self.rpn_head.eval()
            self.semantic_head.eval()
            self.roi_head.img_model.eval()

    def collate(self, points, quantization_mode):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )
    
    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""

    def extract_feats(self, batch_dict, is_train=True):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        
        if is_train:
            points = batch_dict['points']
            feat_3d = batch_dict['feat_3d']
            mask_chunk = batch_dict['mask_chunk']

            points = [
                torch.cat([p, feat, mask], dim=1) if p.shape[1] > 3 else torch.cat(
                    [p, p, feat, torch.unsqueeze(mask, 1)], dim=1) for p, feat, mask in
                zip(points, feat_3d, mask_chunk)]
            
            field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
            batch_dict['field'] = field
            x = field.sparse()
            
            gt_mask_chunk = x.features[:, -1].bool()
            gt_feat_distill = x.features[:, 3:-1]

            x = ME.SparseTensor(
                x.features[:, :3],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            batch_dict['voxel_points'] = x
            x = self.backbone(x)
            batch_dict['backbone_feat'] = x
            batch_dict['gt_mask_chunk'] = gt_mask_chunk
            batch_dict['gt_feat_distill'] = gt_feat_distill
        else:
            points = batch_dict['points']
            points = [torch.cat([p], dim=1) if p.shape[1] > 3 else torch.cat([p, p], dim=1) for p in points]
            field = self.collate(points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            batch_dict['field'] = field
            x = field.sparse()
            x = ME.SparseTensor(
                x.features[:, :3],
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
            batch_dict['voxel_points'] = x
            x = self.backbone(x)
            batch_dict['backbone_feat'] = x
        return batch_dict

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d,
                      feat_3d, mask_chunk, img_calib, img_metas):  # 
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        batch_dict = {}
        batch_dict['points'] = points
        batch_dict['gt_bboxes_3d'] = gt_bboxes_3d
        # batch_dict['gt_labels_3d'] = gt_labels_3d
        batch_dict['gt_labels_3d'] = [torch.zeros_like(tensor) for tensor in gt_labels_3d]
        batch_dict['gt_labels_3d_owp'] = [torch.zeros_like(tensor) for tensor in gt_labels_3d]
        batch_dict['batch_size'] = len(img_metas)
        batch_dict['feat_3d'] = feat_3d
        batch_dict['mask_chunk'] = mask_chunk

        batch_dict['img_metas'] = img_metas

        batch_dict = self.extract_feats(batch_dict)

        batch_dict['semantic_feat'] = self.semantic_head.forward(batch_dict)

        rpn_result, losses = self.rpn_head.forward_train(batch_dict)
        batch_dict.update(rpn_result)
        ######pc distill 
        feature_distill = batch_dict['semantic_feat'].features
        gt_mask_chunk = batch_dict['gt_mask_chunk']
        gt_feat_distill = batch_dict['gt_feat_distill']
        distill_loss = (1 - torch.nn.CosineSimilarity()
                    (feature_distill[gt_mask_chunk], gt_feat_distill[gt_mask_chunk])).mean()
        losses.update(
            dict(distill_loss = self.distill_loss_weight * distill_loss)
        )

        #########
        if  self.train_stage != '1st':
            batch_dict['img_calib'] = img_calib

            roi_losses = self.roi_head.forward(batch_dict)
            losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.
        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        batch_dict = {}
        batch_dict['points'] = points
        batch_dict['img_metas'] = img_metas
        
        batch_dict['text_features_file'] = self.text_features_file
        batch_dict['text_features'] = torch.load(self.text_features_file).to(points[0].device)
        
        batch_dict = self.extract_feats(batch_dict, is_train=False)
        batch_dict['semantic_feat'] = self.semantic_head.forward(batch_dict)
 
        # bbox_list = batch_dict['pred_bbox_list_1st']
        if self.train_stage == '1st':
            rpn_result = self.rpn_head.forward_test_openvoc(batch_dict)
            bbox_list = rpn_result['pred_bbox_list_1st']
        else :
            rpn_result = self.rpn_head.forward_test(batch_dict)
            batch_dict.update(rpn_result)
            # bbox_list = self.roi_head.forward_test(batch_dict)
            bbox_list = self.roi_head.forward_test(batch_dict)
        
        '''
        novel_calss_list = ['paper towel dispenser', 'trash bin', 'coffee machine', 'bottle', 'sink', 'sponge']
        for i, novel_class in enumerate(novel_calss_list):
            import os
            output_prefix = 'output_file_sv/' + novel_class + '_bbox/'
            os.makedirs(output_prefix, exist_ok=True)
            bbox_novel = bbox_list[0][0][bbox_list[0][2]==i].tensor
            print(novel_class, torch.concat([bbox_novel, bbox_list[0][1][bbox_list[0][2]==i].unsqueeze(1)], axis=1))
            #bbox_novel = bbox_novel[0:1, :]
            bbox_novel[:, 2] += bbox_novel[:, 5] / 2
            try:
                
                output_filename = output_prefix + str(img_metas[0]['sample_idx']) + '_' + novel_class + '.ply'
                write_oriented_bbox(bbox_novel.cpu().numpy(), output_filename)
                if str(img_metas[0]['sample_idx']) == 'scene0487_00_000000':
                    print(img_metas[0]['sample_idx'], bbox_list[0][1][bbox_list[0][2]==i][0], novel_class, bbox_novel.shape[0])
                    os.makedirs('/home/yifei/hyf/fcaf3d/output_file_sv/' + img_metas[0]['sample_idx'], exist_ok=True)
                    #np.save('output_file_sv/' + img_metas[0]['sample_idx'] + '/0010_'+novel_class+'.npy', bbox_novel.cpu().numpy())
                    write_oriented_bbox(bbox_novel.cpu().numpy(), 'output_file_sv/' + img_metas[0]['sample_idx'] + '/0017_'+novel_class+'.ply')
            except:
                print(img_metas[0]['sample_idx'], [], novel_class, 0)
                
        if str(img_metas[0]['sample_idx']) == 'scene0487_00_000000':
            
            #print(bbox_list[0][2])
            bbox_all = bbox_list[0][0].tensor
            bbox_all[:, 2] += bbox_all[:, 5] / 2
            #print(bbox_all)
            write_oriented_bbox(bbox_all.cpu().numpy(), 'output_file_sv/' + img_metas[0]['sample_idx'] + '/0017_all.ply') 
        '''

        if len(bbox_list[0]) == 3:
            bbox_results = [
                bbox3d2result(boxes, scores, labels)
                for boxes, scores, labels in bbox_list
            ]
        elif len(bbox_list[0]) == 6:
            bbox_results = [
                bbox3d2result_owp(boxes_ar, scores_ar, labels_ar, boxes_ap, scores_ap, labels_ap,test='owp')
                for boxes_ar, scores_ar, labels_ar, boxes_ap, scores_ap, labels_ap in bbox_list
            ]

        return bbox_results

    def set_epoch(self, epoch):

        self.rpn_head.epoch = epoch

    def set_iter(self, iter):

        self.rpn_head.iter = iter

   

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
