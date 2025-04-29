import argparse
from os import path as osp
import os
from concurrent import futures as futures
import mmcv
import numpy as np
import cv2
from icecream import ic
import shutil
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def img_norm(img):
	imagenet_std = np.array([0.26862954, 0.26130258, 0.27577711])
	imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073])
	img = ((img/255) - imagenet_mean) / imagenet_std
	return img


class ScanNetData_OV3Det(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.dataset_name = 'OVD_sv_data_two_stage_b10'
        if split == "train":
            self.classes = ['object']
            self.cat_ids = np.arange(1)            
        else :
            self.classes = ["toilet", "bed", "chair", "sofa", "dresser", "table", "cabinet", "bookshelf", "pillow", "sink",
                             "bathtub", "refridgerator", "desk", "night stand", "counter", "door", "curtain", "box", "lamp", "bag"]
            self.cat_ids = np.arange(20)
    
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join('./scannet_25k/meta_data',
                              f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list =  sorted(
                list(
                    set([os.path.basename(x)[0:19] for x in os.listdir(self.root_dir) if ("intrinsic" not in x) and ("pc" in x)])
                )
            )
    
        if split == 'train':
            train_pt_path = '/home/yifei/hyf/data/ScanNet_processed/scannet_frames_25k_train'
            train_pt_id = sorted(list(set([os.path.basename(x)[0:19] for x in os.listdir(train_pt_path) if ("intrinsic" not in x) and ("pc" in x)])))
            sample_id_pt_set = set(train_pt_id) & set(self.sample_id_list)
            self.sample_id_list = list(sample_id_pt_set)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, f'{idx}_bbox.npy')
        mmcv.check_file_exist(box_file)
        box = np.load(box_file)
        box[:,3:6] *= 2
        box[:,6] *= -1
        return box

    def get_images(self, idx):
        paths = []
        img_path_original = osp.join('./scannet_frames_25k', idx[:12], 'color', f'{idx[13:]}.jpg')
        path = osp.join('img', f'{self.split}', f'{idx}.jpg')
        if not osp.exists(path):
            shutil.copy(img_path_original, path)
        return path
    
    def get_intrinsics(self, idx):
        path = osp.join(self.root_dir, idx[:-7]+"_image_intrinsic.txt") 
        return  load_matrix_from_txt(path)

    def get_pose(self, idx):
        path = osp.join(self.root_dir, f'{idx}_pose.txt') 
        pose = np.linalg.inv(load_matrix_from_txt(path))
        return  pose
    

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir,
                                    f'{sample_idx}_pc.npy')
            
            points = np.load(pts_filename).astype(np.float32)
            mmcv.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(
                osp.join(self.save_path, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            info['intrinsics'] = self.get_intrinsics(sample_idx)               
            image_path = self.get_images(sample_idx)

            info['image_path'] = image_path
            info['pose'] = self.get_pose(sample_idx)
            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 7
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix =  np.eye(4)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
                print('done')
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)





if __name__ == '__main__':
    data_path_train = "./scannet_25k/ScanNet_processed/scannet_frames_25k_train"
    save_path = "./scannet_25k"
    train_dataset = ScanNetData_OV3Det(root_path=data_path_train, split='train', save_path=save_path)
    infos_train = train_dataset.get_infos(
            num_workers=4, has_label=True)
    train_filename = os.path.join(save_path, 'scannet_infos_train_ov3det.pkl')
    print("train data load scuccess !!")
    mmcv.dump(infos_train, train_filename, 'pkl')
    print('scannet info train file is saved to {train_filename}')
    print(len(infos_train))

    data_path_val = "./scannet_25k/ScanNet_processed/scannet_frames_25k_val"
    save_path = "./scannet_25k"
    val_dataset = ScanNetData_OV3Det(root_path=data_path_val, split='val', save_path=save_path)
    infos_val = val_dataset.get_infos(
            num_workers=4, has_label=True)
    
    val_filename = os.path.join(save_path, 'scannet_infos_val_ov3det.pkl')
    print("val data load scuccess !!")
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'scannet info val file is saved to {val_filename}')
    print(len(infos_val))
    print("all done")