# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import numpy as np
import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose
#from mmdet3d.core import instance_seg_eval, show_result, show_seg_result, instance_seg_eval_v2, detector_instance_eval, show_result_v2
from mmdet3d.core import instance_seg_eval, show_result, show_seg_result, instance_seg_eval_v2, show_result_v2


from icecream import ic
UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        # return confusion[:num_classes, :num_classes]
        return confusion

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate_seg(pred_ids, gt_ids, stdout=False):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture')
    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]
    print(class_accs)
    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou



@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    # CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    #            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    #            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
    #            'garbagebin')
    CLASSES = ('bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
                 'wardrobe')
    # CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    #            'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    #            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
    #            "bag", "bottle", "bowl", "camera", "can",
    #            "cap", "clock", "keyboard", "display", "earphone",
    #            "jar", "knife", "lamp", "laptop", "microphone",
    #            "microwave", "mug", "printer", "remote control", "phone",
    #            "alarm", "book", "cake", "calculator", "candle",
    #            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
    #            "dishes", "doll", "eraser", "eye_glasses", "file_box",
    #            "fork", "fruit", "globe", "hat", "mirror",
    #            "notebook", "pencil", "plant", "plate", "radio",
    #            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
    #            "vase", "vegetables")

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 7
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    # def show(self, results, out_dir, show=True, pipeline=None):
    #     """Results visualization.

    #     Args:
    #         results (list[dict]): List of bounding boxes results.
    #         out_dir (str): Output directory of visualization result.
    #         show (bool): Visualize the results online.
    #         pipeline (list[dict], optional): raw data loading for showing.
    #             Default: None.
    #     """
    #     assert out_dir is not None, 'Expect out_dir, got none.'
    #     pipeline = self._get_pipeline(pipeline)
    #     for i, result in enumerate(results):
    #         data_info = self.data_infos[i]
    #         pts_path = data_info['pts_path']
    #         file_name = osp.split(pts_path)[-1].split('.')[0]
    #         points = self._extract_data(i, pipeline, 'points').numpy()
    #         gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
    #         pred_bboxes = result['boxes_3d'].tensor.numpy()
    #         show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
    #                     show)
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_bboxes, gt_labels,
                           pred_bboxes, pred_labels, out_dir, file_name)

@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetSegDataset(Custom3DSegDataset):
    r"""ScanNet Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    CLASSES = ('wall', 'floor', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
                 'wardrobe')
    # CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    #            'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    #            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
    #            "bag", "bottle", "bowl", "camera", "can",
    #            "cap", "clock", "keyboard", "display", "earphone",
    #            "jar", "knife", "lamp", "laptop", "microphone",
    #            "microwave", "mug", "printer", "remote control", "phone",
    #            "alarm", "book", "cake", "calculator", "candle",
    #            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
    #            "dishes", "doll", "eraser", "eye_glasses", "file_box",
    #            "fork", "fruit", "globe", "hat", "mirror",
    #            "notebook", "pencil", "plant", "plate", "radio",
    #            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
    #            "vase", "vegetables")
    VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    ALL_CLASS_IDS = tuple(range(25))
    # VALID_CLASS_IDS = (3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,
    #                                 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    #                                 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    #                                 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93)
    # ALL_CLASS_IDS = tuple(range(94))

    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
        [82, 84, 163],
        [82, 100, 201],
        [100, 84, 201],
        [201, 84, 100],
    ]

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 modality=None,
                 test_mode=False,
                 ignore_index=None,
                 scene_idxs=None,
                 **kwargs):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            palette=palette,
            modality=modality,
            test_mode=test_mode,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            **kwargs)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=np.max(self.ALL_CLASS_IDS)),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(self.PALETTE), self.ignore_index, show)

    def get_scene_idxs(self, scene_idxs):
        """Compute scene_idxs for data sampling.

        We sample more times for scenes with more points.
        """
        # when testing, we load one whole scene every time
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError(
                'please provide re-sampled scene indexes for training')

        return super().get_scene_idxs(scene_idxs)

    def format_results(self, results, txtfile_prefix=None):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            txtfile_prefix (str): The prefix of saved files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """
        import mmcv

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        mmcv.mkdir_or_exist(txtfile_prefix)

        # need to map network output to original label idx
        pred2label = np.zeros(len(self.VALID_CLASS_IDS)).astype(np.int)
        for original_label, output_idx in self.label_map.items():
            if output_idx != self.ignore_index:
                pred2label[output_idx] = original_label

        outputs = []
        for i, result in enumerate(results):
            info = self.data_infos[i]
            sample_idx = info['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int)
            pred_label = pred2label[pred_sem_mask]
            curr_file = f'{txtfile_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')
            outputs.append(dict(seg_mask=pred_label))

        return outputs, tmp_dir


@DATASETS.register_module()
@SEG_DATASETS.register_module()
class ScanNetInstanceSegDataset(Custom3DSegDataset):
    CLASSES = ('bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
                 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet',
                 'wardrobe')

    VALID_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    # CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    #            'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    #            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
    #            "bag", "bottle", "bowl", "camera", "can",
    #            "cap", "clock", "keyboard", "display", "earphone",
    #            "jar", "knife", "lamp", "laptop", "microphone",
    #            "microwave", "mug", "printer", "remote control", "phone",
    #            "alarm", "book", "cake", "calculator", "candle",
    #            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
    #            "dishes", "doll", "eraser", "eye_glasses", "file_box",
    #            "fork", "fruit", "globe", "hat", "mirror",
    #            "notebook", "pencil", "plant", "plate", "radio",
    #            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
    #            "vase", "vegetables")
    # VALID_CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    # VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39,
    #                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    #                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    #                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93)


    ALL_CLASS_IDS = tuple(range(25))
    # ALL_CLASS_IDS = tuple(range(94))
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:
                - pts_semantic_mask_path (str): Path of semantic masks.
                - pts_instance_mask_path (str): Path of instance masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset. Palette is simply ignored for
        instance segmentation.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
                Defaults to None.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Defaults to None.
        """
        if classes is not None:
            return classes, None
        return self.CLASSES, None

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='PointSegClassMapping',
                valid_cat_ids=self.VALID_CLASS_IDS,
                max_cat_id=40),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'

        load_pipeline = self._get_pipeline(pipeline)
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            raise NotImplementedError('show is not implemented for now')

        return ret_dict

@DATASETS.register_module()
class ScanNetInstanceSegV2Dataset(ScanNetDataset):
    VALID_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        load_pipeline = self._build_default_pipeline()
        pred_instance_masks = [result['instance_mask'] for result in results]
        pred_instance_labels = [result['instance_label'] for result in results]
        pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])
        ret_dict = instance_seg_eval_v2(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.VALID_CLASS_IDS,
            class_labels=self.CLASSES,
            options=options,
            logger=logger)

        if show:
            self.show(results, out_dir)

        return ret_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        assert out_dir is not None, 'Expect out_dir, got none.'
        load_pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_instance_mask, gt_sem_mask = self._extract_data(
                i, load_pipeline, ['points', 'pts_instance_mask', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            gt_inst_mask_final = np.zeros_like(gt_instance_mask)
            for cls_idx in self.VALID_CLASS_IDS:
                mask = gt_sem_mask == cls_idx
                gt_inst_mask_final += mask.numpy()
            gt_instance_mask[gt_inst_mask_final == 0] = -1

            pred_instance_masks = result['instance_mask']
            pred_instance_scores = result['instance_score']

            pred_instance_masks_sort = pred_instance_masks[pred_instance_scores.argsort()]
            pred_instance_masks_label = pred_instance_masks_sort[0].long() - 1
            for i in range(1, pred_instance_masks_sort.shape[0]):
                pred_instance_masks_label[pred_instance_masks_sort[i].bool()] = i

            palette = np.random.random((max(max(pred_instance_masks_label) + 2, max(gt_instance_mask) + 2), 3)) * 255
            palette[-1] = 255

            show_seg_result(points, gt_instance_mask,
                            pred_instance_masks_label, out_dir, file_name,
                            palette)

@DATASETS.register_module()
class ScanNet200InstanceSegDataset(ScanNetInstanceSegV2Dataset):
    VALID_CLASS_IDS = (
        2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
        155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
        488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)

@DATASETS.register_module()
class ScanNetOWPInstanceSegDataset(ScanNetInstanceSegV2Dataset):
    VALID_CLASS_IDS = (1,)

@DATASETS.register_module()
class ScanNetDetInstanceDataset_OWP(ScanNetDataset):
    VALID_CLASS_IDS = (1,)
    # VALID_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    
    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_mask_3d=True,
                with_seg_3d=True),
            dict(
                type='DefaultFormatBundle3D',
                with_label=False,
                class_names=self.CLASSES),
            dict(
                type='Collect3D',
                keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 options=None,
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in instance segmentation protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str]): Metrics to be evaluated.
            options (dict, optional): options for instance_seg_eval.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Defaults to False.
            out_dir (str, optional): Path to save the visualization results.
                Defaults to None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        #from mmdet3d.core.evaluation import detector_instance_eval
        from mmdet3d.core.evaluation import seg_eval

        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        load_pipeline = self._build_default_pipeline()
        # pred_instance_masks = [result['instance_mask'] for result in results]
        # pred_instance_labels = [result['instance_label'] for result in results]
        # pred_instance_scores = [result['instance_score'] for result in results]
        gt_semantic_masks, gt_instance_masks = zip(*[
            self._extract_data(
                index=i,
                pipeline=load_pipeline,
                key=['pts_semantic_mask', 'pts_instance_mask'],
                load_annos=True) for i in range(len(self.data_infos))
        ])

        # ret_dict = instance_seg_eval_v2(
        #     gt_semantic_masks,
        #     gt_instance_masks,
        #     pred_instance_masks,
        #     pred_instance_labels,
        #     pred_instance_scores,
        #     valid_class_ids=self.VALID_CLASS_IDS,
        #     class_labels=self.CLASSES,
        #     options=options,
        #     logger=logger)
       
        pred_sem_masks = [result['semantic_mask'] for result in results]
        pred = torch.cat(pred_sem_masks)
        gt = torch.cat(gt_semantic_masks)
        accumu_iou = evaluate_seg(pred.numpy(),
                                             gt.numpy(),
                                             stdout=True)
        print(accumu_iou)
        seg_dict = seg_eval(
            gt_semantic_masks,
            pred_sem_masks,
            label2cat,
            255,
            logger=logger)
        # out_dir = '/home/ubuntu/xxw/SmallDet/mmdetection3d/work_dirs/dspdet_distill_seg_visualization_0620'
        # self.show(results, out_dir, pipeline)
        return seg_dict
    
        # ret_dict = detector_instance_eval(
        #     gt_annos,
        #     gt_semantic_masks,
        #     gt_instance_masks,
        #     results,
        #     iou_thr,
        #     label2cat,
        #     valid_class_ids=self.VALID_CLASS_IDS,
        #     class_labels=self.CLASSES,
        #     options=options,
        #     logger=logger,
        #     box_type_3d=self.box_type_3d,
        #     box_mode_3d=self.box_mode_3d
        # )

        # if show:
        #     self.show(results, out_dir)

        # return ret_dict

    # def show(self, results, out_dir, show=True, pipeline=None):
    #     assert out_dir is not None, 'Expect out_dir, got none.'
    #     load_pipeline = self._build_default_pipeline()
    #     for i, result in enumerate(results):
    #         data_info = self.data_infos[i]
    #         pts_path = data_info['pts_path']
    #         file_name = osp.split(pts_path)[-1].split('.')[0]
    #         points, gt_instance_mask, gt_sem_mask = self._extract_data(
    #             i, load_pipeline, ['points', 'pts_instance_mask', 'pts_semantic_mask'], load_annos=True)
    #         points = points.numpy()
    #         gt_inst_mask_final = np.zeros_like(gt_instance_mask)
    #         for cls_idx in self.VALID_CLASS_IDS:
    #             mask = gt_sem_mask == cls_idx
    #             gt_inst_mask_final += mask.numpy()
    #         gt_instance_mask[gt_inst_mask_final == 0] = -1

    #         pred_instance_masks = result['instance_mask']
    #         pred_instance_scores = result['instance_score']

    #         pred_instance_masks_sort = pred_instance_masks[pred_instance_scores.argsort()]
    #         pred_instance_masks_label = pred_instance_masks_sort[0].long() - 1
    #         for i in range(1, pred_instance_masks_sort.shape[0]):
    #             pred_instance_masks_label[pred_instance_masks_sort[i].bool()] = i

    #         palette = np.random.random((max(max(pred_instance_masks_label) + 2, max(gt_instance_mask) + 2), 3)) * 255
    #         palette[-1] = 255

    #         show_seg_result(points, gt_instance_mask,
    #                         pred_instance_masks_label, out_dir, file_name,
    #                         palette)
            
    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ]
        pipeline = self._get_pipeline(pipeline)
        
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, gt_sem_mask = self._extract_data(
                i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask,
                            pred_sem_mask, out_dir, file_name,
                            np.array(PALETTE), 255, show)

@DATASETS.register_module()
class ScanNetDataset_DetOWPDistill(Custom3DDataset):
    
    # CLASSES = ('all_object',)
    CLASSES =('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture')
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
        
    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
                        show)
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval
        
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        # from icecream import ic
   
        owp = True
        if metric != None :
            if 'owp' in metric:
                owp = True
            else:
                owp = False
        owp = True
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            owp=owp)
        return ret_dict


@DATASETS.register_module()
class ScanNetDatasetOV3DetSV(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename
        
        if self.modality['use_camera']:
            img_info = dict()
            #img_info['image_path'] = osp.join(self.data_root, info['image_path'])
            img_info['image_path'] = osp.join('/home/yifei/hyf/data/scannet_frames_25k', sample_idx, 'color', info['image_path'][0])   
            img_info['intrinsic'] = info['intrinsics']
            #img_info['pose'] = info['pose']
            img_info['pose'] = np.loadtxt(osp.join('/home/yifei/hyf/data/scannet_frames_25k', sample_idx, 'pose', info['image_path'][0].split('.')[0]+'.txt'))
            input_dict['img_info'] = img_info
        
        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos
        #     if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
        #         return None
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
            return None
        return input_dict
        

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # pts_instance_mask_path = osp.join(self.data_root,
        #                                   info['pts_instance_mask_path'])
        # pts_semantic_mask_path = osp.join(self.data_root,
        #                                   info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # pts_instance_mask_path=pts_instance_mask_path,
            # pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        
        # input_dict['ann_info'] = dict(
        #     axis_align_matrix=self._get_axis_align_matrix(
        #         self.data_infos[index]))  
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)
    
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        # from icecream import ic
        # ic(metric)
        owp = False
        if metric != None :
            if 'owp' in metric:
                owp = True
            else:
                owp = False
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            owp=owp)

        # out_dir = '/home/ubuntu/xxw/SmallDet/mmdetection3d/work_dirs/debug_visualize_owp'
        # self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
    

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        print('showing!!!!!')
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            # gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to('cpu').numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            # pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_bboxes = torch.cat((pred_bboxes.gravity_center, pred_bboxes.tensor[:, 3:]), dim=1).to('cpu').numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_bboxes, gt_labels,
                           pred_bboxes, pred_labels, out_dir, file_name)
            

@DATASETS.register_module()
class ScanNetDatasetOV3DetSVPC(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        print('data root:', data_root)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename
        
        if self.modality['use_camera']:
            img_info = dict()
            img_info['image_path'] = []
            #print(info.keys())
            for image_filename in info['image_path']:
                scene_name = image_filename.split('.')[0]               
                image_path = osp.join(self.data_root, scene_name, image_filename)
                img_info['image_path'].append(image_path)   
            img_info['intrinsic'] = info['intrinsics']
            img_info['pose'] = info['pose']
            input_dict['img_info'] = img_info
        
        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos
        #     if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
        #         return None
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
            return None
        return input_dict
        

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        # pts_instance_mask_path = osp.join(self.data_root,
        #                                   info['pts_instance_mask_path'])
        # pts_semantic_mask_path = osp.join(self.data_root,
        #                                   info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # pts_instance_mask_path=pts_instance_mask_path,
            # pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        #print(example)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        
        # input_dict['ann_info'] = dict(
        #     axis_align_matrix=self._get_axis_align_matrix(
        #         self.data_infos[index]))  
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)
    
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}

        # from icecream import ic
        # ic(metric)
        owp = False
        if metric != None :
            if 'owp' in metric:
                owp = True
            else:
                owp = False
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)

        # out_dir = '/home/ubuntu/xxw/SmallDet/mmdetection3d/work_dirs/debug_visualize_owp'
        # self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
    

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        print('showing!!!!!')
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            # gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_bboxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to('cpu').numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            # pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_bboxes = torch.cat((pred_bboxes.gravity_center, pred_bboxes.tensor[:, 3:]), dim=1).to('cpu').numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_bboxes, gt_labels,
                           pred_bboxes, pred_labels, out_dir, file_name)