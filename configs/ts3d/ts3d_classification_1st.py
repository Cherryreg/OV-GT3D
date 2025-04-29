#find_unused_parameters = True
voxel_size = .01
train_stage = '1st'   ####train:['1st', '2st', '3st']   val：['1st', '2st', '3st', '3st_loc']
model = dict(
    type='MinkTwoStage3DDetectorPC2OBJ_V2',
    voxel_size=voxel_size,
    train_stage=train_stage,
    backbone=dict(type='MinkResNet_DetInsseg', in_channels=3, max_channels=128, depth=34,  pool=False, norm='batch'),
    rpn_head=dict(
        type='RPNHead_TS3D',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
        n_reg_outs=6,
        n_classes=1,
        voxel_size=voxel_size,
        pts_prune_threshold=100000,
        first_assigner=dict(
            type='TS3DInstanceAssigner_DetInsseg',
            top_pts_threshold=6),
        cls_loss=dict(type='FocalLoss', loss_weight=0.25),),
    semantic_head=dict(
        type='SemanticHead_TS3D',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
        n_feat=768
        ),
    roi_head=dict(
        type='ROIHead_TS3D',
        in_channels=768,
        out_channels=768,
        voxel_size=voxel_size,
        n_classes=1,
        n_reg_outs=6,
        train_stage=train_stage,
        assigner=dict(
            type='ProposalTargetLayerV2',
            roi_per_image=128,
            fg_ratio=0.6,
            reg_fg_thresh=0.3,
            cls_fg_thresh=0.55,
            cls_bg_thresh=0.15,
            cls_bg_thresh_l0=0.1,
            hard_bg_ratio=0.6,),
        roi_conv_kernel=5,
        grid_size=7,
        coord_key=1,
        code_size=6,
        use_center_pooling=True,
        use_simple_pooling=True,
        bbox_loss=dict(
            type='SmoothL1Loss', reduction='none', loss_weight=1.0),), 
    text_features_file = 'data/text_features/OIS3D_text_features_prompt.pt',
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=500,  # 600  1200
        iou_thr=.8,   #0.7
        score_thr=0.0,   #9
        test_score_thr=0.0,
        test_iou_thr=0.5,
        test_nms_pre=2000,
        owp=True,
        train_stage=train_stage,
        output_box=False,)
        )

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

lr_config = dict(policy='step', warmup=None, step=[8, 11])

if train_stage == '1st':
    runner = dict(type='EpochBasedRunner', max_epochs=12)
    n_repeat = 10
elif train_stage == '2st':
    runner = dict(type='EpochBasedRunner', max_epochs=6)
    n_repeat = 10
elif train_stage == '3st':
    runner = dict(type='EpochBasedRunner', max_epochs=6)
    n_repeat = 10
elif train_stage == '3st_loc':
    runner = dict(type='EpochBasedRunner', max_epochs=12)
    n_repeat = 10
#runner = dict(type='IterBasedRunner', max_iters=10)


#runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
checkpoint_config = dict(interval=1, max_keep_ckpts=40)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
if train_stage == '1st':
    load_from = None
elif train_stage == '2st':
    load_from =  None
elif train_stage == '3st':
    load_from = None
elif train_stage == '3st_loc':
    load_from = None
resume_from = None
workflow = [('train', 1)]


n_points = 100000
dataset_type = 'ScanNetDatasetRec2SV' 
data_root_train = 'data/scannet_all/'
data_root_val = "data/scannet_OIS3D/"
class_names = ('all_objet', )

class_names = ('cabinet','bed','chair','sofa','table','door','counter','desk','sink',
                'bathtub','window','bookshelf','curtain','refridgerator','toilet','picture','shower')

if train_stage == '1st':
    train_pipeline = [
    dict(
        type= 'LoadMultiviewFeaturesPointsFromFile',  #'LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        datapath_feat='data/scannet/scannet_multiview_openseg'),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointFeatSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.087266, 0.087266],
            scale_ratio_range=[.9, 1.1],
            translation_std=[.1, .1, .1],
            shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'feat_3d', 'mask_chunk',
            'gt_from_image_mask', 'image', 'intrinsics', 'extrinsics' ])
    ]
else:
    train_pipeline = [
    dict(
        type= 'LoadMultiviewFeaturesPointsFromFile',  #'LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2],
        datapath_feat='data/scannet/scannet_multiview_openseg'),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointFeatSample', num_points=n_points),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'feat_3d', 'mask_chunk',
            'gt_from_image_mask', 'image', 'intrinsics', 'extrinsics' ])
    ]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # we do not sample 100k points for scannet, as very few scenes have
            # significantly more then 100k points. so it doesn't affect inference
            # time and we ca accept all points
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=n_repeat,
        dataset=dict(
            type=dataset_type,
            data_root=data_root_train,
            ann_file=data_root_train + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file=data_root_val + 'scannet_infos_val.pkl',   #scannet_infos_val.pkl_ov3det
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file=data_root_val + 'scannet_infos_val.pkl',#_ov3det
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
