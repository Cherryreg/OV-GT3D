## Preparing ScanNet V2 Reconstructed Point Cloud Dataset

**Step 1.** Download ScanNet v2 and ScanNet 25k data [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scans' folder under 'data/scannet', and the 'scannet_frames_25k' under 'data/scannet_25k'.

**Step 2.** Download ScanNet multi-view fused OpenSeg features [HERE](https://github.com/pengsongyou/openscene/blob/0f369bc73d0724ae24b5e46bbada193f8ee9d193/scripts/download_fused_features.sh). Link or move the 'scannet_multiview_openseg' folder under 'data/scannet'.

```shell
sh  download_fused_features.sh  ds_id=0
```

**Step 3.** Extract ScanNet data with annotated 3D bounding boxes from 10 classes and create .pkl file for training
```shell
python scannet/scannet_seen10/batch_load_scannet_data.py
python scannet/scannet_seen10/get_gt_image_from_pc.py
python text_features/clip_text_features_all_object.py
python tools/create_data.py scannet --root-path scannet/scannet_seen10 --out-dir scannet/scannet_seen10 --extra-tag scannet
```

**Step 4.** Extract ScanNet data with annotated 3D bounding boxes from 17 classes (OIS3D) and create .pkl file for training
```shell
python scannet/scannet_OIS3D/batch_load_scannet_data.py
python scannet/scannet_OIS3D/get_gt_image_from_pc.py
python text_features/clip_text_features_OIS3D.py
python tools/create_data.py scannet --root-path scannet/scannet_OIS3D --out-dir scannet/scannet_OIS3D --extra-tag scannet
```

**Step 5.** Extract ScanNet data with unannotated 3D bounding boxes and create .pkl file for testing
```shell
python scannet/scannet_all/batch_load_scannet_data.py
python scannet/scannet_all/get_gt_image_from_pc.py
python tools/create_data.py scannet --root-path scannet/scannet_all --out-dir scannet/scannet_all --extra-tag scannet
```

The directory structure after pre-processing should be as below
```
scannet_all (or scannet_seen10, scannet_OIS3D)
├── gt_images
│   ├── scene0000_00_gtbox_image.npy
│   └── ...
├── instance_mask
│   ├── scene0000_00.bin
│   └── ...
├── points
│   ├── scene0000_00.bin
│   └── ...
├── scannet_instance_data
│   ├── scene0000_00_aligned_bbox.npy
│   ├── scene0000_00_axis_align_matrix.npy
│   ├── scene0000_00_ins_label.npy
│   ├── scene0000_00_sem_label.npy
│   ├── scene0000_00_unaligned_bbox.npy
│   ├── scene0000_00_vert.npy
│   └── ...
├── semantic_mask
│   ├── scene0000_00.bin
│   └── ...
├── scannet_infos_train.pkl
└── scannet_infos_val.pkl
```

## Preparing ScanNet V2 Single-view Dataset
**Step 1.** Download processed single-view dataset from [HERE](https://github.com/lyhdet/OV-3DET#Dataset%20preparation), unzip it and move or link the 'ScanNet_processed' folder under 'data/scannet_25k'.

**Step 2.** Extract the single-view data with annotated 3D bounding boxes and create .pkl file for training
```shell
python scannet_25k/generate_pkl_OV3Det.py
python text_features/clip_text_features_OV3Det.py
```

**Step 3.** We followed [OpenScene](https://github.com/pengsongyou/openscene) to extract the single-view features. Please install [OpenScene](https://github.com/pengsongyou/openscene) and run:
```shell
python openscene/scripts/preprocess/preprocess_2d_scannet.py --scannet_path data/scannet/scans --output_path data/scannet_25k/scannet_openseg_sv --label_map_file data/scannet/meta_data/scannetv2-labels.combined.tsv
```

The directory structure after pre-processing should be as below
```
OV3Det
├── img
│   ├── train
│   │   ├── scene0000_00_000000.jpg
│   │   └── ...
│   └── val
│   │   ├── scene0000_01_000000.jpg
│   │   └── ...
├── points
│   ├── scene0000_00_000000.bin
│   └── ...
├── scannet_infos_train_ov3det.pkl
└── scannet_infos_val_ov3det.pkl
```

## Preparing SUN RGB-D Dataset
Please refer to [HERE](https://github.com/open-mmlab/mmdetection3d/tree/main/data/sunrgbd) to generate the SUN RGB-D dataset.

The directory structure after pre-processing should be as below

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   └── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   └── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   └── val_data_idx.txt
├── points
├── sunrgbd_infos_train.pkl
└── sunrgbd_infos_val.pkl
```