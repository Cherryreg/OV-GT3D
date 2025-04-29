import numpy as np

# try:
#     import MinkowskiEngine as ME
# except ImportError:
#     import warnings
#     warnings.warn(
#         'Please follow `getting_started.md` to install MinkowskiEngine.`')
try:
    import MinkowskiEngine as ME
    from MinkowskiTensorField import TensorField
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
    pass

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn
from scipy.spatial import ConvexHull, Delaunay
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
import clip
import random
import torchvision.transforms as T
from PIL import Image
try:
	from torchvision.transforms import InterpolationMode
	BICUBIC = InterpolationMode.BICUBIC
except ImportError:
	BICUBIC = Image.BICUBIC

from mmdet3d.models import build_backbone, build_head
from mmdet.core.bbox.builder import build_assigner
from mmcv.ops import diff_iou_rotated_3d
from icecream import ic
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import axis_aligned_bbox_overlaps_3d
import pdb
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.utils.ovd_box_coder import OVDResidualCoder
import torch.nn.functional as F
#from .ovd_dtcc_head import GenericMLP

class SimplePoolingLayer(nn.Module):
    def __init__(self, channels=[128,128,128], grid_kernel_size = 5, grid_num = 7, voxel_size=0.04, coord_key=2,
                    point_cloud_range=[-5.12*3, -5.12*3, -5.12*3, 5.12*3, 5.12*3, 5.12*3], # simply use a large range
                    corner_offset_emb=False, pooling=False):
        super(SimplePoolingLayer, self).__init__()
        # build conv
        self.voxel_size = voxel_size
        self.coord_key = coord_key
        grid_size = [int((point_cloud_range[3] - point_cloud_range[0])/voxel_size), 
                     int((point_cloud_range[4] - point_cloud_range[1])/voxel_size), 
                     int((point_cloud_range[5] - point_cloud_range[2])/voxel_size)]
        self.grid_size = grid_size
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_num = grid_num
        self.pooling = pooling
        self.count = 0
        self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3)
        self.grid_bn = ME.MinkowskiBatchNorm(channels[1])
        self.grid_relu = ME.MinkowskiELU()
        if self.pooling:
            self.pooling_conv = ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=grid_num, dimension=3)
            self.pooling_bn = ME.MinkowskiBatchNorm(channels[1])

        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.grid_conv.kernel, std=.01)
        if self.pooling:
            nn.init.normal_(self.pooling_conv.kernel, std=.01)

    def forward(self, sp_tensor, grid_points, grid_corners=None, box_centers=None, batch_size=None):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: bxnum_roisx216, 4 (b,x,y,z)
            grid_corners (optional): bxnum_roisx216, 8, 3
            box_centers: bxnum_rois, 4 (b,x,y,z)
        """
        grid_coords = grid_points.long()
        grid_coords[:, 1:4] = torch.floor(grid_points[:, 1:4] / self.voxel_size) # get coords (grid conv center)
        grid_coords[:, 1:4] = torch.clamp(grid_coords[:, 1:4], min=-self.grid_size[0] / 2 + 1, max=self.grid_size[0] / 2 - 1) # -192 ~ 192
        grid_coords_positive = grid_coords[:, 1:4] + self.grid_size[0] // 2 
        merge_coords = grid_coords[:, 0] * self.scale_xyz + \
                        grid_coords_positive[:, 0] * self.scale_yz + \
                        grid_coords_positive[:, 1] * self.scale_z + \
                        grid_coords_positive[:, 2] 
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        # unq_grid_coords = torch.stack((unq_coords // self.scale_xyz,
        #                             (unq_coords % self.scale_xyz) // self.scale_yz,
        #                             (unq_coords % self.scale_yz) // self.scale_z,
        #                             unq_coords % self.scale_z), dim=1) 
        unq_grid_coords = torch.stack((torch.div(unq_coords, self.scale_xyz, rounding_mode='trunc'),
                                    torch.div((unq_coords % self.scale_xyz), self.scale_yz, rounding_mode='trunc'),
                                    torch.div((unq_coords % self.scale_yz), self.scale_z, rounding_mode='trunc'),
                                    unq_coords % self.scale_z), dim=1) 
        unq_grid_coords[:, 1:4] -= self.grid_size[0] // 2
        unq_grid_coords[:, 1:4] *= self.coord_key
        unq_grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, unq_grid_coords.int()))) 
        unq_features = unq_grid_sp_tensor.F
        unq_coords = unq_grid_sp_tensor.C
        new_features = unq_features[unq_inv]

        if self.pooling:
            # fake grid
            fake_grid_coords = torch.ones(self.grid_num, self.grid_num, self.grid_num, device=unq_grid_coords.device)
            fake_grid_coords = torch.nonzero(fake_grid_coords) - self.grid_num // 2 
            fake_grid_coords = fake_grid_coords.unsqueeze(0).repeat(grid_coords.shape[0] // fake_grid_coords.shape[0], 1, 1) 
            # fake center
            fake_centers = fake_grid_coords.new_zeros(fake_grid_coords.shape[0], 3) 
            fake_batch_idx = torch.arange(fake_grid_coords.shape[0]).to(fake_grid_coords.device) 
            fake_center_idx = fake_batch_idx.reshape([-1, 1])
            fake_center_coords = torch.cat([fake_center_idx, fake_centers], dim=-1).int() 
            
            fake_grid_idx = fake_batch_idx.reshape([-1, 1, 1]).repeat(1, fake_grid_coords.shape[1], 1) 
            fake_grid_coords = torch.cat([fake_grid_idx, fake_grid_coords], dim=-1).reshape([-1, 4]).int()

            grid_sp_tensor = ME.SparseTensor(coordinates=fake_grid_coords, features=new_features)
            pooled_sp_tensor = self.pooling_conv(grid_sp_tensor, fake_center_coords) 
            pooled_sp_tensor = self.pooling_bn(pooled_sp_tensor) 
            return pooled_sp_tensor.F
        else:
            return new_features

class SimplePoolingLayer_3st(nn.Module):
    def __init__(self, channels=[128,128,128], grid_kernel_size = 5, grid_num = 7, voxel_size=0.04, coord_key=2,
                    point_cloud_range=[-5.12*3, -5.12*3, -5.12*3, 5.12*3, 5.12*3, 5.12*3], # simply use a large range
                    corner_offset_emb=False, pooling=False):
        super(SimplePoolingLayer_3st, self).__init__()
        # build conv
        self.voxel_size = voxel_size
        self.coord_key = coord_key
        grid_size = [int((point_cloud_range[3] - point_cloud_range[0])/voxel_size), 
                     int((point_cloud_range[4] - point_cloud_range[1])/voxel_size), 
                     int((point_cloud_range[5] - point_cloud_range[2])/voxel_size)]
        self.grid_size = grid_size
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        self.grid_num = grid_num
        self.pooling = pooling
        self.count = 0
        self.grid_conv = ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=grid_kernel_size, dimension=3)
        self.grid_bn = ME.MinkowskiBatchNorm(channels[1])
        self.grid_relu = ME.MinkowskiELU()
        if self.pooling:
            self.pooling_conv = ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=grid_num, dimension=3)
            self.pooling_bn = ME.MinkowskiBatchNorm(channels[1])
            self.pooling_relu = ME.MinkowskiELU()
            self.cls_conv = ME.MinkowskiConvolution(channels[2], 1, kernel_size=1, bias=True, dimension=3)
            self.iou_conv = ME.MinkowskiConvolution(channels[2], 1, kernel_size=1, bias=True, dimension=3)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.grid_conv.kernel, std=.01)
        if self.pooling:
            nn.init.normal_(self.pooling_conv.kernel, std=.01)
            nn.init.normal_(self.cls_conv.kernel, std=.01)
            nn.init.normal_(self.iou_conv.kernel, std=.01)

    def forward(self, sp_tensor, grid_points, grid_corners=None, box_centers=None, batch_size=None):
        """
        Args:
            sp_tensor: minkowski tensor
            grid_points: bxnum_roisx216, 4 (b,x,y,z)
            grid_corners (optional): bxnum_roisx216, 8, 3
            box_centers: bxnum_rois, 4 (b,x,y,z)
        """
        grid_coords = grid_points.long()
        grid_coords[:, 1:4] = torch.floor(grid_points[:, 1:4] / self.voxel_size) # get coords (grid conv center)
        grid_coords[:, 1:4] = torch.clamp(grid_coords[:, 1:4], min=-self.grid_size[0] / 2 + 1, max=self.grid_size[0] / 2 - 1) # -192 ~ 192
        grid_coords_positive = grid_coords[:, 1:4] + self.grid_size[0] // 2 
        merge_coords = grid_coords[:, 0] * self.scale_xyz + \
                        grid_coords_positive[:, 0] * self.scale_yz + \
                        grid_coords_positive[:, 1] * self.scale_z + \
                        grid_coords_positive[:, 2] 
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        # unq_grid_coords = torch.stack((unq_coords // self.scale_xyz,
        #                             (unq_coords % self.scale_xyz) // self.scale_yz,
        #                             (unq_coords % self.scale_yz) // self.scale_z,
        #                             unq_coords % self.scale_z), dim=1) 
        unq_grid_coords = torch.stack((torch.div(unq_coords, self.scale_xyz, rounding_mode='trunc'),
                                    torch.div((unq_coords % self.scale_xyz), self.scale_yz, rounding_mode='trunc'),
                                    torch.div((unq_coords % self.scale_yz), self.scale_z, rounding_mode='trunc'),
                                    unq_coords % self.scale_z), dim=1) 
        unq_grid_coords[:, 1:4] -= self.grid_size[0] // 2
        unq_grid_coords[:, 1:4] *= self.coord_key
        unq_grid_sp_tensor = self.grid_relu(self.grid_bn(self.grid_conv(sp_tensor, unq_grid_coords.int()))) 
        unq_features = unq_grid_sp_tensor.F
        unq_coords = unq_grid_sp_tensor.C
        new_features = unq_features[unq_inv]

        if self.pooling:
            # fake grid
            fake_grid_coords = torch.ones(self.grid_num, self.grid_num, self.grid_num, device=unq_grid_coords.device)
            fake_grid_coords = torch.nonzero(fake_grid_coords) - self.grid_num // 2 
            fake_grid_coords = fake_grid_coords.unsqueeze(0).repeat(grid_coords.shape[0] // fake_grid_coords.shape[0], 1, 1) 
            # fake center
            fake_centers = fake_grid_coords.new_zeros(fake_grid_coords.shape[0], 3) 
            fake_batch_idx = torch.arange(fake_grid_coords.shape[0]).to(fake_grid_coords.device) 
            fake_center_idx = fake_batch_idx.reshape([-1, 1])
            fake_center_coords = torch.cat([fake_center_idx, fake_centers], dim=-1).int() 
            
            fake_grid_idx = fake_batch_idx.reshape([-1, 1, 1]).repeat(1, fake_grid_coords.shape[1], 1) 
            fake_grid_coords = torch.cat([fake_grid_idx, fake_grid_coords], dim=-1).reshape([-1, 4]).int()

            grid_sp_tensor = ME.SparseTensor(coordinates=fake_grid_coords, features=new_features)
            pooled_sp_tensor = self.pooling_conv(grid_sp_tensor, fake_center_coords) 
            pooled_sp_tensor = self.pooling_bn(pooled_sp_tensor) 
            x_conv = self.pooling_relu(pooled_sp_tensor)
            cls_sp_tensor = self.cls_conv(x_conv)
            iou_sp_tensor = self.iou_conv(x_conv)
            return pooled_sp_tensor.F, cls_sp_tensor.F, iou_sp_tensor.F
        else:
            return new_features


@HEADS.register_module()
class ROIHead_TS3D(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                voxel_size,
                n_classes,
                n_reg_outs,
                assigner,
                roi_conv_kernel=5,
                grid_size=7,
                coord_key=2,
                code_size=7,
                use_center_pooling=True,
                use_simple_pooling=True,
                bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                cls_loss=dict(type='FocalLoss', reduction='none'),
                iou_loss=dict(type='CrossEntropyLoss', use_sigmoid=True),
                train_stage='2st',
                train_cfg=None,
                test_cfg=None,):
        super(ROIHead_TS3D, self).__init__(init_cfg=None)
        self.train_stage = train_stage
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_class = n_classes
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.iou_loss = build_loss(iou_loss)
        self.use_center_pooling = use_center_pooling
        self.use_simple_pooling = use_simple_pooling
        self.grid_size = grid_size
        self.mlps = [[768,768,768]]
        self.code_size = code_size
        self.voxel_size = voxel_size
        self.box_coder = OVDResidualCoder(code_size=code_size, encode_angle_by_sincos=False)
        
        self._init_layers(in_channels, out_channels, roi_conv_kernel, grid_size, voxel_size, coord_key)


        self.img_model = self.build_img_encoder()

        self.text = ["A photo of human",
                    "A photo of sneakers",
                    "A photo of chair",
                    "A photo of hat",
                    "A photo of lamp",
                    "A photo of bottle",
                    "A photo of cabinet/shelf",
                    "A photo of cup",
                    "A photo of car",
                    "A photo of glasses",
                    "A photo of picture/frame",
                    "A photo of desk",
                    "A photo of handbag",
                    "A photo of street lights",
                    "A photo of book",
                    "A photo of plate",
                    "A photo of helmet",
                    "A photo of leather shoes",
                    "A photo of pillow",
                    "A photo of glove",
                    "A photo of potted plant",
                    "A photo of bracelet",
                    "A photo of flower",
                    "A photo of monitor",
                    "A photo of storage box",
                    "A photo of plants pot/vase",
                    "A photo of bench",
                    "A photo of wine glass",
                    "A photo of boots",
                    "A photo of dining table",
                    "A photo of umbrella",
                    "A photo of boat",
                    "A photo of flag",
                    "A photo of speaker",
                    "A photo of trash bin/can",
                    "A photo of stool",
                    "A photo of backpack",
                    "A photo of sofa",
                    "A photo of belt",
                    "A photo of carpet",
                    "A photo of basket",
                    "A photo of towel/napkin",
                    "A photo of slippers",
                    "A photo of bowl",
                    "A photo of barrel/bucket",
                    "A photo of coffee table",
                    "A photo of suv",
                    "A photo of toy",
                    "A photo of tie",
                    "A photo of bed",
                    "A photo of traffic light",
                    "A photo of pen/pencil",
                    "A photo of microphone",
                    "A photo of sandals",
                    "A photo of canned",
                    "A photo of necklace",
                    "A photo of mirror",
                    "A photo of faucet",
                    "A photo of bicycle",
                    "A photo of bread",
                    "A photo of high heels",
                    "A photo of ring",
                    "A photo of van",
                    "A photo of watch",
                    "A photo of combine with bowl",
                    "A photo of sink",
                    "A photo of horse",
                    "A photo of fish",
                    "A photo of apple",
                    "A photo of traffic sign",
                    "A photo of camera",
                    "A photo of candle",
                    "A photo of stuffed animal",
                    "A photo of cake",
                    "A photo of motorbike/motorcycle",
                    "A photo of wild bird",
                    "A photo of laptop",
                    "A photo of knife",
                    "A photo of cellphone",
                    "A photo of paddle",
                    "A photo of truck",
                    "A photo of cow",
                    "A photo of power outlet",
                    "A photo of clock",
                    "A photo of drum",
                    "A photo of fork",
                    "A photo of bus",
                    "A photo of hanger",
                    "A photo of nightstand",
                    "A photo of pot/pan",
                    "A photo of sheep",
                    "A photo of guitar",
                    "A photo of traffic cone",
                    "A photo of tea pot",
                    "A photo of keyboard",
                    "A photo of tripod",
                    "A photo of hockey stick",
                    "A photo of fan",
                    "A photo of dog",
                    "A photo of spoon",
                    "A photo of blackboard/whiteboard",
                    "A photo of balloon",
                    "A photo of air conditioner",
                    "A photo of cymbal",
                    "A photo of mouse",
                    "A photo of telephone",
                    "A photo of pickup truck",
                    "A photo of orange",
                    "A photo of banana",
                    "A photo of airplane",
                    "A photo of luggage",
                    "A photo of skis",
                    "A photo of soccer",
                    "A photo of trolley",
                    "A photo of oven",
                    "A photo of remote",
                    "A photo of combine with glove",
                    "A photo of paper towel",
                    "A photo of refrigerator",
                    "A photo of train",
                    "A photo of tomato",
                    "A photo of machinery vehicle",
                    "A photo of tent",
                    "A photo of shampoo/shower gel",
                    "A photo of head phone",
                    "A photo of lantern",
                    "A photo of donut",
                    "A photo of cleaning products",
                    "A photo of sailboat",
                    "A photo of tangerine",
                    "A photo of pizza",
                    "A photo of kite",
                    "A photo of computer box",
                    "A photo of elephant",
                    "A photo of toiletries",
                    "A photo of gas stove",
                    "A photo of broccoli",
                    "A photo of toilet",
                    "A photo of stroller",
                    "A photo of shovel",
                    "A photo of baseball bat",
                    "A photo of microwave",
                    "A photo of skateboard",
                    "A photo of surfboard",
                    "A photo of surveillance camera",
                    "A photo of gun",
                    "A photo of Life saver",
                    "A photo of cat",
                    "A photo of lemon",
                    "A photo of liquid soap",
                    "A photo of zebra",
                    "A photo of duck",
                    "A photo of sports car",
                    "A photo of giraffe",
                    "A photo of pumpkin",
                    "A photo of Accordion/keyboard/piano",
                    "A photo of radiator",
                    "A photo of converter",
                    "A photo of tissue ",
                    "A photo of carrot",
                    "A photo of washing machine",
                    "A photo of vent",
                    "A photo of cookies",
                    "A photo of cutting/chopping board",
                    "A photo of tennis racket",
                    "A photo of candy",
                    "A photo of skating and skiing shoes",
                    "A photo of scissors",
                    "A photo of folder",
                    "A photo of baseball",
                    "A photo of strawberry",
                    "A photo of bow tie",
                    "A photo of pigeon",
                    "A photo of pepper",
                    "A photo of coffee machine",
                    "A photo of bathtub",
                    "A photo of snowboard",
                    "A photo of suitcase",
                    "A photo of grapes",
                    "A photo of ladder",
                    "A photo of pear",
                    "A photo of american football",
                    "A photo of basketball",
                    "A photo of potato",
                    "A photo of paint brush",
                    "A photo of printer",
                    "A photo of billiards",
                    "A photo of fire hydrant",
                    "A photo of goose",
                    "A photo of projector",
                    "A photo of sausage",
                    "A photo of fire extinguisher",
                    "A photo of extension cord",
                    "A photo of facial mask",
                    "A photo of tennis ball",
                    "A photo of chopsticks",
                    "A photo of Electronic stove and gas stove",
                    "A photo of pie",
                    "A photo of frisbee",
                    "A photo of kettle",
                    "A photo of hamburger",
                    "A photo of golf club",
                    "A photo of cucumber",
                    "A photo of clutch",
                    "A photo of blender",
                    "A photo of tong",
                    "A photo of slide",
                    "A photo of hot dog",
                    "A photo of toothbrush",
                    "A photo of facial cleanser",
                    "A photo of mango",
                    "A photo of deer",
                    "A photo of egg",
                    "A photo of violin",
                    "A photo of marker",
                    "A photo of ship",
                    "A photo of chicken",
                    "A photo of onion",
                    "A photo of ice cream",
                    "A photo of tape",
                    "A photo of wheelchair",
                    "A photo of plum",
                    "A photo of bar soap",
                    "A photo of scale",
                    "A photo of watermelon",
                    "A photo of cabbage",
                    "A photo of router/modem",
                    "A photo of golf ball",
                    "A photo of pine apple",
                    "A photo of crane",
                    "A photo of fire truck",
                    "A photo of peach",
                    "A photo of cello",
                    "A photo of notepaper",
                    "A photo of tricycle",
                    "A photo of toaster",
                    "A photo of helicopter",
                    "A photo of green beans",
                    "A photo of brush",
                    "A photo of carriage",
                    "A photo of cigar",
                    "A photo of earphone",
                    "A photo of penguin",
                    "A photo of hurdle",
                    "A photo of swing",
                    "A photo of radio",
                    "A photo of CD",
                    "A photo of parking meter",
                    "A photo of swan",
                    "A photo of garlic",
                    "A photo of french fries",
                    "A photo of horn",
                    "A photo of avocado",
                    "A photo of saxophone",
                    "A photo of trumpet",
                    "A photo of sandwich",
                    "A photo of cue",
                    "A photo of kiwi fruit",
                    "A photo of bear",
                    "A photo of fishing rod",
                    "A photo of cherry",
                    "A photo of tablet",
                    "A photo of green vegetables",
                    "A photo of nuts",
                    "A photo of corn",
                    "A photo of key",
                    "A photo of screwdriver",
                    "A photo of globe",
                    "A photo of broom",
                    "A photo of pliers",
                    "A photo of hammer",
                    "A photo of volleyball",
                    "A photo of eggplant",
                    "A photo of trophy",
                    "A photo of board eraser",
                    "A photo of dates",
                    "A photo of rice",
                    "A photo of tape measure/ruler",
                    "A photo of dumbbell",
                    "A photo of hamimelon",
                    "A photo of stapler",
                    "A photo of camel",
                    "A photo of lettuce",
                    "A photo of goldfish",
                    "A photo of meat balls",
                    "A photo of medal",
                    "A photo of toothpaste",
                    "A photo of antelope",
                    "A photo of shrimp",
                    "A photo of rickshaw",
                    "A photo of trombone",
                    "A photo of pomegranate",
                    "A photo of coconut",
                    "A photo of jellyfish",
                    "A photo of mushroom",
                    "A photo of calculator",
                    "A photo of treadmill",
                    "A photo of butterfly",
                    "A photo of egg tart",
                    "A photo of cheese",
                    "A photo of pomelo",
                    "A photo of pig",
                    "A photo of race car",
                    "A photo of rice cooker",
                    "A photo of tuba",
                    "A photo of crosswalk sign",
                    "A photo of papaya",
                    "A photo of hair dryer",
                    "A photo of green onion",
                    "A photo of chips",
                    "A photo of dolphin",
                    "A photo of sushi",
                    "A photo of urinal",
                    "A photo of donkey",
                    "A photo of electric drill",
                    "A photo of spring rolls",
                    "A photo of tortoise/turtle",
                    "A photo of parrot",
                    "A photo of flute",
                    "A photo of measuring cup",
                    "A photo of shark",
                    "A photo of steak",
                    "A photo of poker card",
                    "A photo of binoculars",
                    "A photo of llama",
                    "A photo of radish",
                    "A photo of noodles",
                    "A photo of mop",
                    "A photo of yak",
                    "A photo of crab",
                    "A photo of microscope",
                    "A photo of barbell",
                    "A photo of Bread/bun",
                    "A photo of baozi",
                    "A photo of lion",
                    "A photo of red cabbage",
                    "A photo of polar bear",
                    "A photo of lighter",
                    "A photo of mangosteen",
                    "A photo of seal",
                    "A photo of comb",
                    "A photo of eraser",
                    "A photo of pitaya",
                    "A photo of scallop",
                    "A photo of pencil case",
                    "A photo of saw",
                    "A photo of table tennis  paddle",
                    "A photo of okra",
                    "A photo of starfish",
                    "A photo of monkey",
                    "A photo of eagle",
                    "A photo of durian",
                    "A photo of rabbit",
                    "A photo of game board",
                    "A photo of french horn",
                    "A photo of ambulance",
                    "A photo of asparagus",
                    "A photo of hoverboard",
                    "A photo of pasta",
                    "A photo of target",
                    "A photo of hotair balloon",
                    "A photo of chainsaw",
                    "A photo of lobster",
                    "A photo of iron",
                    "A photo of flashlight",
                    "A photo of unclear image",
                    ######new class\
                    "A photo of counter",
                    "A photo of curtain",
                    "A photo of bookshelf",
                    "A photo of shower"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text = clip.tokenize(self.text).to(device)
        self.text_feats = self.batch_encode_text(text)
        self.text_num = self.text_feats.shape[0] -1 
        self.text_label = torch.arange(self.text_num, dtype=torch.int).to(device)
        ##ov3det_20c
        '''
        self.eval_text = ["A photo of toilet",
                    "A photo of bed",
                    "A photo of chair",
                    "A photo of sofa",
                    "A photo of dresser",
                    "A photo of table",
                    "A photo of cabinet",
                    "A photo of bookshelf",
                    "A photo of pillow",
                    "A photo of sink",
                    "A photo of bathtub",
                    "A photo of refridgerator",
                    "A photo of desk",
                    "A photo of night stand",
                    "A photo of counter",
                    "A photo of door",
                    "A photo of curtain",
                    "A photo of box",
                    "A photo of lamp",
                    "A photo of bag",
                    'A photo of ground',
                    'A photo of wall',
                    'A photo of floor',
                    "An unclear image",
                    "A photo of background",
                    "There is no object in this image",]
        '''
        ####ois3d_17c
        self.eval_text = ["A photo of cabinet",
                    "A photo of bed",
                    "A photo of chair",
                    "A photo of sofa",
                    "A photo of table",
                    "A photo of door",
                    "A photo of counter",
                    "A photo of desk",
                    "A photo of sink",
                    "A photo of bathtub",
                    "A photo of window",
                    "A photo of bookshelf",
                    "A photo of curtain",
                    "A photo of refridgerator",
                    "A photo of toilet",
                    "A photo of picture",
                    "A photo of shower",
                    'A photo of ground',
                    'A photo of wall',
                    'A photo of floor',
                    "An unclear image",
                    "A photo of background",
                    "There is no object in this image",]
        # ##B10N7
        # self.eval_text = ["A photo of bed",
        #             "A photo of chair",
        #             "A photo of table",
        #             "A photo of bookeshelf",
        #             "A photo of picture",
        #             "A photo of sink",
        #             "A photo of bathtub",
        #             'A photo of ground',
        #             'A photo of wall',
        #             'A photo of floor',
        #             "An unclear image",
        #             "A photo of background",
        #             "There is no object in this image",]
        eval_text = clip.tokenize(self.eval_text).to(device)
        self.eval_text_feats = self.batch_encode_text(eval_text)

        self.eval_text_num = self.eval_text_feats.shape[0] - 6
        self.eval_text_label = torch.arange(self.eval_text_num, dtype=torch.int).to(device)


        self.init_weights()
        

    
    def _init_layers(self, in_channels, out_channels, roi_conv_kernel, grid_size, voxel_size, coord_key, kernel_size=1):
        if self.train_stage == '2st':
            self.clip_header = nn.ModuleList()
            for i in range(len(self.mlps)): # different feature source, default only use semantic feature
                mlp = self.mlps[i] 
                pool_layer = SimplePoolingLayer(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                                voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=True)
                self.clip_header.append(pool_layer)

            # self.clip_header_linear = nn.Linear(out_channels, 768, bias=True)

        elif self.train_stage == '3st':
            self.clip_header = nn.ModuleList()
            for i in range(len(self.mlps)): # different feature source, default only use semantic feature
                mlp = self.mlps[i] 
                pool_layer = SimplePoolingLayer(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                                voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=True)
                self.clip_header.append(pool_layer)
                
            self.roi_grid_pool_layers_3st = nn.ModuleList()
            for i in range(len(self.mlps)): # different feature source, default only use semantic feature
                mlp = self.mlps[i] 
                pool_layer = SimplePoolingLayer_3st(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                                voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=self.use_center_pooling)
                self.roi_grid_pool_layers_3st.append(pool_layer)  

        elif self.train_stage == '3st_loc':
            self.roi_grid_pool_layers_3st = nn.ModuleList()
            for i in range(len(self.mlps)): # different feature source, default only use semantic feature
                mlp = self.mlps[i] 
                pool_layer = SimplePoolingLayer_3st(channels=mlp, grid_kernel_size=roi_conv_kernel, grid_num=grid_size, \
                                                voxel_size=voxel_size*coord_key, coord_key=coord_key, pooling=self.use_center_pooling)
                self.roi_grid_pool_layers_3st.append(pool_layer)

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for n, m in self.named_modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            if '3st' in n:
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """

        if self.train_stage == '2st':
            batch_dict_2st = self._forward_2st(batch_dict)
            losses = batch_dict_2st['obj_distill_loss']
        elif self.train_stage == '3st' or self.train_stage == '3st_loc':
            batch_dict_3st = self._forward_3st(batch_dict)
            losses = batch_dict_3st['roi_loss']
        return losses
    
    def forward_test(self,batch_dict):
        if self.train_stage == '2st':
            bbox_list = self.forward_test_2st(batch_dict)
        elif self.train_stage == '3st':
            bbox_list = self.forward_test_3st(batch_dict) 
        elif self.train_stage == '3st_loc':
            bbox_list = self.forward_test_3st_loc(batch_dict) 
        return bbox_list
    
    def forward_test_2st(self,batch_dict):
        points = batch_dict['points']
        img_metas = batch_dict['img_metas']
        pred_boxes_3d = batch_dict['pred_bbox_list_2st']
        batch_dict['batch_size'] = len(points)

        #####obj
        batch_dict['rois_ovd'] = [pred_boxes_3d[0][0]]
        pred_bboxes_feats_clip_header = self.roi_grid_pool_ovd(batch_dict)
        ##res
        seg_feats = batch_dict['semantic_feat']
      
        bbox_maxplooing_feats = self.maxpooling_bbox_feat_v2(seg_feats, batch_dict['rois_ovd'], img_metas)
        bbox_maxplooing_feats = torch.cat(bbox_maxplooing_feats, dim=0)
        pred_bboxes_feats_clip_header =  pred_bboxes_feats_clip_header + bbox_maxplooing_feats
        #######
        ##norm
        pred_bboxes_feats_clip_header = pred_bboxes_feats_clip_header / (pred_bboxes_feats_clip_header.norm(dim=1,keepdim=True) + 1e-5)
        ######

        eval_text_output_before_clip_header = self.eval_text_feats
        eval_text_output = self.eval_text_feats / self.eval_text_feats.norm(dim=1, keepdim=True)
        eval_text_feat_label = {"text_feat": eval_text_output[:self.eval_text_num, :], "text_label": self.eval_text_label}
        
        bbox_preds = pred_boxes_3d[0][0][:,:6]
        box_score_preds = pred_boxes_3d[0][1]
        pc_sem_cls_logits, pc_objectness_prob, pc_sem_cls_prob, pc_all_label = self.classify_pc(pred_bboxes_feats_clip_header, eval_text_output, self.eval_text_num)
        bbox_list = self._get_bboxesV2(bbox_preds, pc_sem_cls_prob, box_score_preds, img_metas)
        return bbox_list
    
    def forward_test_3st(self,batch_dict):
        points = batch_dict['points']
        x = batch_dict['backbone_feat']
        img_metas = batch_dict['img_metas']
        pred_boxes_3d = batch_dict['pred_bbox_list']
        # rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d)
        rois, roi_scores, roi_labels, batch_size= self.reoder_rois_for_refining(pred_boxes_3d)
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels
        batch_dict['batch_size'] = batch_size

        # roi pooling
        pooled_features, roi_pred_bbox_score, roi_pred_bbox_iou = self.roi_grid_pool(batch_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        ##2*fc
        # share_feat = self.shared_fc_layer(pooled_features)
        share_feat = pooled_features
        batch_dict['batch_roi_share_feat'] = share_feat.view(batch_size, -1, share_feat.shape[1])
        ##fc_head
        batch_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou
        batch_dict['roi_pred_bbox_score'] = roi_pred_bbox_score
        
        ####Insseg
        roi_dict = dict()
        roi_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou.view(batch_size, -1, 1)
        roi_dict['roi_pred_bbox_score'] = roi_pred_bbox_score.view(batch_size, -1, 1)
        roi_dict['rois'] = batch_dict['rois']
        roi_dict['roi_scores'] = batch_dict['roi_scores']
        roi_dict['batch_size'] = batch_size

        results_nms = self.get_boxes_3st(roi_dict, img_metas) 
        #####obj
        batch_dict['rois_ovd'] = [results_nms[0][0]]
        pred_bboxes_feats_clip_header = self.roi_grid_pool_ovd(batch_dict)
        ##res
        seg_feats = batch_dict['semantic_feat']
        bbox_maxplooing_feats = self.maxpooling_bbox_feat_v2(seg_feats, batch_dict['rois_ovd'], img_metas)
        bbox_maxplooing_feats = torch.cat(bbox_maxplooing_feats, dim=0)
        pred_bboxes_feats_clip_header =  pred_bboxes_feats_clip_header + bbox_maxplooing_feats
        #######
        ##norm
        pred_bboxes_feats_clip_header = pred_bboxes_feats_clip_header / (pred_bboxes_feats_clip_header.norm(dim=1,keepdim=True) + 1e-5)
        ######

        eval_text_output_before_clip_header = self.eval_text_feats
        eval_text_output = self.eval_text_feats / self.eval_text_feats.norm(dim=1, keepdim=True)
        eval_text_feat_label = {"text_feat": eval_text_output[:self.eval_text_num, :], "text_label": self.eval_text_label}
        
        bbox_preds = results_nms[0][0][:,:6]
        box_score_preds = results_nms[0][1]
        pc_sem_cls_logits, pc_objectness_prob, pc_sem_cls_prob, pc_all_label = self.classify_pc(pred_bboxes_feats_clip_header, eval_text_output, self.eval_text_num)

        bbox_list = self._get_bboxesV2(bbox_preds, pc_sem_cls_prob, box_score_preds, img_metas)

            
        return bbox_list
    
    def forward_test_3st_loc(self,batch_dict):
        
        img_metas = batch_dict['img_metas']
        pred_boxes_3d = batch_dict['pred_bbox_list']
        # rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d)
        rois, roi_scores, roi_labels, batch_size= self.reoder_rois_for_refining(pred_boxes_3d)
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels
        batch_dict['batch_size'] = batch_size

        # roi pooling
        pooled_features, roi_pred_bbox_score, roi_pred_bbox_iou = self.roi_grid_pool(batch_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        ##2*fc
        # share_feat = self.shared_fc_layer(pooled_features)
        share_feat = pooled_features
        batch_dict['batch_roi_share_feat'] = share_feat.view(batch_size, -1, share_feat.shape[1])
        ##fc_head
        batch_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou
        batch_dict['roi_pred_bbox_score'] = roi_pred_bbox_score

        bbox_list = self.get_boxes_3st_loc(batch_dict, img_metas) 
        return bbox_list
    
    def _forward_2st(self, batch_dict):
        
        gt_bboxes_3d = batch_dict['gt_bboxes_3d']
        gt_labels_3d = batch_dict['gt_labels_3d_owp']

        gt_from_image_mask = batch_dict['gt_from_image_mask']
        
        img_metas = batch_dict['img_metas']
        pred_boxes_3d = batch_dict['pred_bbox_list']

        gt_bboxes_3d_tensor = []
        for bs in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_tensor.append(torch.cat((gt_bboxes_3d[bs].gravity_center, gt_bboxes_3d[bs].tensor[:, 3:]), dim=1).to(gt_labels_3d[0].device))


        ##object_distill
        second_pred_bboxes, preds_assign_gt = self.bbox_select_train(pred_boxes_3d, gt_bboxes_3d_tensor)
        ###Delete boxes without image corresponding; mapping images  extrinsics to preds_assign_gt
        extrinsics = batch_dict['extrinsics']
        intrinsics = batch_dict['intrinsics']
        images = batch_dict['image']
        for bs in range(len(second_pred_bboxes)):   
            #mapping
            extrinsics[bs] = extrinsics[bs][preds_assign_gt[bs]]
            images[bs] = images[bs][preds_assign_gt[bs]]
            ##delete
            second_pred_bboxes_mask = gt_from_image_mask[bs][preds_assign_gt[bs]]
            second_pred_bboxes[bs] = second_pred_bboxes[bs][second_pred_bboxes_mask]
            extrinsics[bs] = extrinsics[bs][second_pred_bboxes_mask]
            images[bs] = images[bs][second_pred_bboxes_mask]
        batch_dict['rois_ovd'] = second_pred_bboxes
        pred_bboxes_feats_clip_header = self.roi_grid_pool_ovd(batch_dict)
        ##res
        seg_feats = batch_dict['semantic_feat']
        bbox_maxplooing_feats = self.maxpooling_bbox_feat_v2(seg_feats, second_pred_bboxes, img_metas)
        bbox_maxplooing_feats = torch.cat(bbox_maxplooing_feats, dim=0)
        pred_bboxes_feats_clip_header =  pred_bboxes_feats_clip_header + bbox_maxplooing_feats
        #######
        ##norm
        pred_bboxes_feats_clip_header = pred_bboxes_feats_clip_header / (pred_bboxes_feats_clip_header.norm(dim=1,keepdim=True) + 1e-5)
        
        points = batch_dict['points']
        second_points = points
        second_img_metas = img_metas

        text_output_before_clip_header = self.text_feats
        text_output = self.text_feats / self.text_feats.norm(dim=1, keepdim=True)
        text_feat_label = {"text_feat": text_output[:self.text_num, :], "text_label": self.text_label}

        second_points = torch.stack(second_points, dim=0).detach().cpu().numpy()
        pred_bboxes_corners_aug = self.compute_all_bbox_corners(second_pred_bboxes)
        pred_bboxes = self.restore_aug(second_pred_bboxes, second_img_metas)
        pred_bboxes_corners = self.compute_all_bbox_corners(pred_bboxes)

        ####per per box to mapping 2d 
        patches = []
        for bs in range(len(pred_bboxes_corners)):
            cur_bs_intrinsics = intrinsics[bs]
            cur_bs_extrinsics = extrinsics[bs]
            cur_bs_second_points = second_points[bs]
            cur_bs_images = images[bs]
            for n in range(pred_bboxes_corners[bs].shape[0]):
                pred_bbox_corner_3d = pred_bboxes_corners[bs][n] 
                pred_bbox_corner_2d = self.proj_pointcloud_into_image(pred_bbox_corner_3d, cur_bs_extrinsics[n], cur_bs_intrinsics)
                pred_bbox_corner_aug = pred_bboxes_corners_aug[bs][n]
                pred_bbox = second_pred_bboxes[bs][n]
                patch, valid_3d_bbox = self.crop_per_patch(cur_bs_images[n], pred_bbox_corner_2d, pred_bbox_corner_aug, cur_bs_second_points, pred_bbox)
                patches.append(patch.permute(2,0,1))

        patches = torch.stack(patches)
        pair_img_cnt = patches.shape[0]
        img_output = self.img_model.encode_image(patches)
        # ###分批提取特征
        # batch_patch = 5
        # cur_start = 0
        # cur_end = 0
        # img_output = []
        # while cur_end < pair_img_cnt:
		# 	# print(cur_end)
        #     cur_start = cur_end
        #     cur_end += batch_patch
        #     if cur_end >= pair_img_cnt:
        #         cur_end = pair_img_cnt
         
        #     cur_patch = patches[cur_start:cur_end,::]
        #     cur_patch_feats = self.img_model.encode_image(cur_patch).detach()
        #     img_output.append(cur_patch_feats)

        # img_output = torch.cat(img_output, dim=0)

        if torch.isnan(img_output).any():
            ic('pred_bboxes_corners_2d is None')
        pair_img_output_before_clip_header = img_output[:pair_img_cnt, :]

        img_output = img_output / img_output.norm(dim=1, keepdim=True)

        pair_img_output = img_output[:pair_img_cnt, :]
        
        pair_img_porb, pair_img_label = self.classify(pair_img_output_before_clip_header, text_output_before_clip_header)
        
        pc_feat_label={"pc_feat": pred_bboxes_feats_clip_header,
                                "pc_label": pair_img_label,
                                "pc_prob": pair_img_porb,
                                }

        pair_feat_label={"pair_img_feat": pair_img_output,
                                "pair_img_label": pair_img_label,
                                "pair_img_prob": pair_img_porb,
                                }
        pc_dtcc_loss, pair_dtcc_loss = self.dtcc_pc_img_text(pair_feat_label, pc_feat_label, text_feat_label)
        dtcc_loss_weight = 1      #(self.epoch + 1) /12  
        obj_distill_loss = (dict(
                pc_dtcc_loss=dtcc_loss_weight * pc_dtcc_loss,
                pair_dtcc_loss=dtcc_loss_weight * pair_dtcc_loss))
        batch_dict['obj_distill_loss'] = obj_distill_loss
        return batch_dict
    
    def _forward_3st(self, batch_dict):
        gt_bboxes_3d = batch_dict['gt_bboxes_3d']
        gt_labels_3d = batch_dict['gt_labels_3d_owp']  
        pred_boxes_3d = batch_dict['pred_bbox_list']
        ############2st_loc_roi
        # rois, roi_scores, roi_labels, batch_size = self.reoder_rois_for_refining(pred_boxes_3d)
        rois, roi_scores, roi_labels, batch_size= self.reoder_rois_for_refining(pred_boxes_3d)
        gt_bboxes_3d_tensor = []
        for bs in range(len(gt_bboxes_3d)):
            gt_bboxes_3d_tensor.append(torch.cat((gt_bboxes_3d[bs].gravity_center, gt_bboxes_3d[bs].tensor[:, 3:]), dim=1).to(gt_labels_3d[0].device))

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels
        batch_dict['batch_size'] = batch_size
        batch_dict['gt_bboxes_3d_tensor'] = gt_bboxes_3d_tensor
        #pdb.set_trace()
        

        ####Insseg
        '''
        # roi pooling
        pooled_features, roi_pred_bbox_score, roi_pred_bbox_iou = self.roi_grid_pool(batch_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        ##2*fc
        # share_feat = self.shared_fc_layer(pooled_features)
        share_feat = pooled_features
        batch_dict['batch_roi_share_feat'] = share_feat.view(batch_size, -1, share_feat.shape[1])
        ##fc_head
        batch_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou
        batch_dict['roi_pred_bbox_score'] = roi_pred_bbox_score

        roi_dict = dict()
        roi_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou.view(batch_size, -1, 1)
        roi_dict['roi_pred_bbox_score'] = roi_pred_bbox_score.view(batch_size, -1, 1)
        roi_dict['rois'] = batch_dict['rois']
        roi_dict['roi_scores'] = batch_dict['roi_scores']
        roi_dict['batch_size'] = batch_size
        img_metas = batch_dict['img_metas']
        results_nms = self.get_boxes_3st(roi_dict, img_metas)
        batch_dict['roi_bbox_list'] = results_nms

        '''
        

        # assign targets
        targets_dict = self.assign_targets(batch_dict)
        batch_dict.update(targets_dict)

        # roi pooling
        pooled_features, roi_pred_bbox_score, roi_pred_bbox_iou = self.roi_grid_pool(batch_dict)  
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)  
        ##2*fc
        # share_feat = self.shared_fc_layer(pooled_features)
        share_feat = pooled_features
        batch_dict['batch_roi_share_feat'] = share_feat.view(batch_size, -1, share_feat.shape[1])
        ##fc_head
        batch_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou
        batch_dict['roi_pred_bbox_score'] = roi_pred_bbox_score

        roi_dict = dict()
        roi_dict['roi_pred_bbox_iou'] = roi_pred_bbox_iou.view(batch_size, -1, 1)
        roi_dict['roi_pred_bbox_score'] = roi_pred_bbox_score.view(batch_size, -1, 1)
        roi_dict['rois'] = batch_dict['rois']
        roi_dict['roi_scores'] = batch_dict['roi_scores']
        roi_dict['batch_size'] = batch_size
        img_metas = batch_dict['img_metas']
        results_nms = self.get_boxes_3st(roi_dict, img_metas)
        roi_bbox_list = []
        for i in range(batch_size):
            bboxes = img_metas[0]['box_type_3d'](
            results_nms[i][0][:,:6],
            box_dim=6,
            with_yaw=False,
            origin=(.5, .5, .5))
            roi_bbox_list.append(bboxes)
        batch_dict['roi_bbox_list'] = roi_bbox_list


        roi_loss = self.roi_loss(batch_dict)
        batch_dict['roi_loss'] = roi_loss
        return batch_dict
        
    def _get_bboxesV2(self, bbox_preds, cls_preds,  box_score, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_singleV2(
                bbox_preds=bbox_preds,
                cls_preds=cls_preds,
                box_score=box_score,
                img_meta=img_metas[i])
            results.append(result)
        return results
    def _get_bboxes_singleV2(self, bbox_preds, cls_preds, box_score, img_meta):
        box_scores = box_score.unsqueeze(1)
        bbox_preds = bbox_preds
        scores = cls_preds * box_score
        boxes_ap, scores_ap, labels_ap = self._nms_openvoc(bbox_preds, scores, 0.0, 0.5, img_meta)
        boxes_insseg, scores_insseg, labels_insseg = self._nms_openvoc_insseg(bbox_preds, scores, 0.0, 0.5, img_meta)

        '''
        if len(boxes_ap) > self.test_cfg.test_roi_nms_pre:
            _, idxs = scores_ap.topk(self.test_cfg.test_roi_nms_pre)
            #pdb.set_trace()
            boxes_ap = boxes_ap[idxs]
            scores_ap = scores_ap[idxs]
            labels_ap = labels_ap[idxs]
        '''
        #pdb.set_trace()
        return boxes_ap, scores_ap, labels_ap, boxes_insseg, scores_insseg, labels_insseg
    
    def get_boxes_3st(self, input_dict, img_meta):
        
        batch_size = input_dict['batch_size']
        # semantic_feat = input_dict['bbox_maxplooing_feats']
        
        rcnn_cls = input_dict['roi_pred_bbox_score']
        rcnn_iou = input_dict['roi_pred_bbox_iou']
        roi_scores = input_dict['roi_scores']
        batch_box_preds = input_dict['rois']  
        results = []
        for bs_id in range(batch_size):
            # nms
            boxes = batch_box_preds[bs_id]
            theta = 0.4   ##0.8 越高可能效果越好
            scores = pow(roi_scores[bs_id].unsqueeze(1), theta) * pow(rcnn_cls[bs_id].sigmoid() * rcnn_iou[bs_id].sigmoid(), (1 - theta))
            ###
            max_scores, _ = scores.max(dim=1) 
            if len(scores) > 300 > 0:
                _, ids = max_scores.topk(300)
                boxes = boxes[ids]
                scores = scores[ids]
            results.append((boxes,scores))
        return results
    
    def get_boxes_3st_loc(self, input_dict, img_meta):
        
        batch_size = input_dict['batch_size']
        # semantic_feat = input_dict['bbox_maxplooing_feats']
        
        rcnn_cls = input_dict['roi_pred_bbox_score']
        rcnn_iou = input_dict['roi_pred_bbox_iou']
        roi_scores = input_dict['roi_scores']
        batch_box_preds = input_dict['rois']  
        results = []
        for bs_id in range(batch_size):
            # nms
            boxes = batch_box_preds[bs_id]
            theta = 0.4   
            scores = pow(roi_scores[bs_id].unsqueeze(1), theta) * pow(rcnn_cls.sigmoid() * rcnn_iou.sigmoid(), (1 - theta))
            ###
            max_scores, _ = scores.max(dim=1) 
            if len(scores) > 300 > 0:
                _, ids = max_scores.topk(300)
                boxes = boxes[ids]
                scores = scores[ids]
            result = self._nms_openvoc(boxes, scores, 0.0, 0.8, img_meta)
            results.append(result)

        return results
    
    ###maxpooling
    def maxpooling_bbox_feat_v2(self, seg_feats, bboxes, img_metas):
        results = []
        
        for i in range(len(img_metas)):
            coordinates, features = seg_feats.decomposed_coordinates_and_features
            coordinates = coordinates[0]
            features = features[0]
            rois = (bboxes[i])[:,:6]
            n_points = len(coordinates)
            n_boxes = len(rois)
            points = coordinates * self.voxel_size * seg_feats.tensor_stride[0]
            points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
            rois = rois.unsqueeze(0).expand(n_points, n_boxes, 6)
            # ic(rois.shape[1])
            face_distances = self._get_face_distances(points, rois)
            inside_box_condition = face_distances.min(dim=-1).values > 0
            mask_list = torch.unbind(inside_box_condition, dim=1)
            box_feat = []
            for i in range(len(mask_list)):
                mask_ = mask_list[i]
                if mask_.sum() > 0:
                    #point_feat  max pooling-->  box_feat  -->   box_label
                    feat_ = features[mask_]                
                    pooled_feature = torch.mean(feat_, dim=0)
                else:
                    pooled_feature = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
                box_feat.append(pooled_feature)
            results.append(torch.stack(box_feat))
            torch.cuda.empty_cache()
        return results

#####two_stage roi crop 
    def reoder_rois_for_refining(self, pred_boxes_3d):
        """
        Args:
            pred_boxes_3d: List[(box, score, label), (), ...]
        """
        batch_size = len(pred_boxes_3d)
        num_max_rois = max([len(preds[0]) for preds in pred_boxes_3d])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_boxes_3d[0][0]

        if len(pred_boxes_3d[0]) == 4:
            use_sem_score = True
        else:
            use_sem_score = False

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        if use_sem_score:
            roi_sem_scores = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes_3d[0][3].shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(pred_boxes_3d[bs_idx][0])            
            rois[bs_idx, :num_boxes, :] = pred_boxes_3d[bs_idx][0]
            roi_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][1]
            roi_labels[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][2]
            if use_sem_score:
                roi_sem_scores[bs_idx, :num_boxes] = pred_boxes_3d[bs_idx][3]
        
        if use_sem_score:
            return rois, roi_scores, roi_labels, roi_sem_scores, batch_size
        else:
            return rois, roi_scores, roi_labels, batch_size

    def assign_targets(self, input_dict):
        with torch.no_grad():
            targets_dict = self.assigner(input_dict)
        batch_size = input_dict['batch_size']
        rois = targets_dict['rois'] # b, num_max_rois, 7
        gt_of_rois = targets_dict['gt_of_rois'] # b, num_max_rois, 7
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        gt_label_of_rois = targets_dict['gt_label_of_rois'] # b, num_max_rois
        
        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        roi_ry = rois[:, :, 6] 
        # also change gt angle to 0 ~ 2pi
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry # 0 - 0 = 0

        targets_dict['gt_of_rois'] = gt_of_rois

        return targets_dict
    
    def roi_grid_pool(self, input_dict):
        """
        Args:
            input_dict:
                rois: b, num_max_rois, 7
                batch_size: b
                middle_feature_list: List[mink_tensor]
        """
        rois = input_dict['rois']
        batch_size = input_dict['batch_size']
        middle_feature_list = input_dict['semantic_feat']
        if not isinstance(middle_feature_list, list):
            middle_feature_list = [middle_feature_list]
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  

        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        batch_idx = rois.new_zeros(batch_size, roi_grid_xyz.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx

        pooled_features_list = []
        cls_scores, iou_scores = [], []
        for k, cur_sp_tensors in enumerate(middle_feature_list):
            pool_layer = self.roi_grid_pool_layers_3st[k]
            if self.use_simple_pooling:
                batch_grid_points = torch.cat([batch_idx, roi_grid_xyz], dim=-1) 
                batch_grid_points = batch_grid_points.reshape([-1, 4])
                new_features, cls_score, iou_score = pool_layer(cur_sp_tensors, grid_points=batch_grid_points)
            else:
                raise NotImplementedError
            pooled_features_list.append(new_features)
            cls_scores.append(cls_score)
            iou_scores.append(iou_score)
        ms_pooled_feature = torch.cat(pooled_features_list, dim=-1)
        ms_cls_scores = torch.cat(cls_scores, dim=-1)
        ms_iou_scores = torch.cat(iou_scores, dim=-1)
        return ms_pooled_feature, ms_cls_scores, ms_iou_scores
    
    def roi_grid_pool_ovd(self, input_dict):
        """
        Args:
            input_dict:
                rois: b, num_max_rois, 7
                batch_size: b
                middle_feature_list: List[mink_tensor]
        """
        rois = input_dict['rois_ovd']
        batch_size = input_dict['batch_size']
        middle_feature_list = input_dict['semantic_feat']
        if not isinstance(middle_feature_list, list):
            middle_feature_list = [middle_feature_list]
        batch_roi_grid_xyz, _ = self.get_global_grid_points_of_roi_ovd(
            rois, grid_size=self.grid_size
        )  
        
        batch_ms_pooled_feature = []
        for bs in range(batch_size):
            batch_roi_grid_xyz[bs] = batch_roi_grid_xyz[bs].view(1, -1, 3)
            batch_idx = batch_roi_grid_xyz[bs].new_ones(1, batch_roi_grid_xyz[bs].shape[1], 1) * bs
            grid_points = torch.cat([batch_idx, batch_roi_grid_xyz[bs]], dim=-1) 
            grid_points = grid_points.reshape([-1, 4])
            pooled_features_list = []

            for k, cur_sp_tensors in enumerate(middle_feature_list):
                pool_layer = self.clip_header[k]
                new_features = pool_layer(cur_sp_tensors, grid_points=grid_points)
                # new_features = self.clip_header_linear(new_features)
                pooled_features_list.append(new_features)
            pooled_feature = torch.cat(pooled_features_list, dim=-1)
            batch_ms_pooled_feature.append(pooled_feature)

        ms_pooled_feature = torch.cat(batch_ms_pooled_feature, dim=0)
        return ms_pooled_feature
    
    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1]) 
        batch_size_rcnn = rois.shape[0]
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size) 
        if self.code_size > 6:
            global_roi_grid_points = rotate_points_along_z(
                local_roi_grid_points.clone(), rois[:, 6]
            ).squeeze(dim=1)
        else:
            global_roi_grid_points = local_roi_grid_points

        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = global_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points
    
    def get_global_grid_points_of_roi_ovd(self, second_pred_bboxes, grid_size):
        batch_global_roi_grid_points, batch_local_roi_grid_points = [],[]
        for bs in range(len(second_pred_bboxes)):
            rois = second_pred_bboxes[bs]
            batch_size_rcnn = rois.shape[0]
            local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size) 
            # if self.code_size > 6:
            #     global_roi_grid_points = rotate_points_along_z(
            #         local_roi_grid_points.clone(), rois[:, 6]
            #     ).squeeze(dim=1)
            # else:
            global_roi_grid_points = local_roi_grid_points

            global_center = rois[:, 0:3].clone()
            global_roi_grid_points = global_roi_grid_points + global_center.unsqueeze(dim=1)

            batch_global_roi_grid_points.append(global_roi_grid_points)
            batch_local_roi_grid_points.append(local_roi_grid_points)
        
        return batch_global_roi_grid_points, batch_local_roi_grid_points
    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  
        
        return roi_grid_points
    
    def roi_loss(self, batch_dict):
        roi_loss_dict = {}

        rcnn_loss_score, tb_dict = self.get_box_cls_layer_loss(batch_dict)
        roi_loss_dict.update(tb_dict)


        rcnn_loss_iou, iou_tb_dict = self.get_box_iou_layer_loss(batch_dict)
        roi_loss_dict.update(iou_tb_dict)
        return roi_loss_dict
    
    def get_box_cls_layer_loss(self, forward_ret_dict):
        
        rcnn_cls = forward_ret_dict['roi_pred_bbox_score']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        rcnn_cls_flat = rcnn_cls.view(-1)
        batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
        # cls_valid_mask = (rcnn_cls_labels >= 0).float()
        cls_valid_mask = (rcnn_cls_labels > 0).float()
        rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        rcnn_loss_cls = rcnn_loss_cls  ##* self.loss_weight.RCNN_CLS_WEIGHT
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls}
        return rcnn_loss_cls, tb_dict
    
    def get_box_iou_layer_loss(self, forward_ret_dict):
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou = forward_ret_dict['roi_pred_bbox_iou']  # (rcnn_batch_size, C)
        rcnn_batch_roi_ious_gt = forward_ret_dict['batch_roi_ious'].view(-1)
        fg_mask = rcnn_cls_labels > 0
        fg_sum = fg_mask.long().sum().item()
        tb_dict = {}
        loss_iou = self.iou_loss(rcnn_iou[fg_mask], rcnn_batch_roi_ious_gt[fg_mask].unsqueeze(1), avg_factor=fg_sum)
        tb_dict['rcnn_loss_iou'] = loss_iou

        return loss_iou, tb_dict
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
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds, roi_labels=None, gt_bboxes_3d=None, gt_labels_3d=None, roi_sem_scores=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.code_size
        batch_cls_preds = None

        batch_box_preds = box_preds.view(batch_size, -1, code_size) 

        # decode box
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()[..., :code_size]
        local_rois[:, :, 0:3] = 0
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)
        
        if self.code_size > 6:
            roi_ry = rois[:, :, 6].view(-1)
            batch_box_preds = rotate_points_along_z(
                batch_box_preds.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz 
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)

        return batch_cls_preds, batch_box_preds

    
    
    def roi_nms(self, bboxes, scores, labels, img_meta):
        n_classes = self.num_class
        yaw_flag = bboxes.shape[1] == 7
        

        max_scores, _ = scores.max(dim=1)
        if len(scores) > self.test_cfg.test_nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.test_nms_pre)
            bboxes = bboxes[ids]
            scores = scores[ids]
            labels = labels[ids]

        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            if scores.ndim == 2:
                # ids = (scores[:, i] > self.test_score_thr) & (bboxes.sum() != 0) # reclass
                ids = (labels == i) & (scores[:, i] > self.test_cfg.test_score_thr) & (bboxes.sum() != 0) # no reclass
            else:
                ids = (labels == i) & (scores > self.test_cfg.test_score_thr) & (bboxes.sum() != 0)
            if not ids.any():
                continue
            class_scores = scores[ids] if scores.ndim == 1 else scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.test_iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

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
        # ic(scores.shape, nms_scores.shape)
        return nms_bboxes, nms_scores, nms_labels
    
    def _nms_openvoc(self, bboxes, scores, score_thr, iou_thr, img_meta):
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
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels
    
    def _nms_openvoc_insseg(self, bboxes, scores, score_thr, iou_thr, img_meta):
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
        n_classes = 1
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
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


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
        shift = shift.permute(1, 0, 2)
        centers = boxes[..., :3] + shift
        dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
        dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
        dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)
    

    ####ovd
    def compute_all_bbox_corners(self, pred_bboxes):

        batch_size = len(pred_bboxes)
        all_corners = []
        for cur_bs in range(batch_size):
            cur_bs_corners = []
            if torch.is_tensor(pred_bboxes[cur_bs]):
                pred_bbox = pred_bboxes[cur_bs].detach().cpu().numpy()
            else:
                pred_bbox = pred_bboxes[cur_bs]
            for cur_bbox in range(pred_bbox.shape[0]):
                cur_center = pred_bbox[cur_bbox, :3]
                cur_size = pred_bbox[cur_bbox, 3:6] / 2

                cur_heading = pred_bbox[cur_bbox, 6]
                cur_corners = self.my_compute_box_3d(cur_center, cur_size, cur_heading)
                cur_bs_corners.append(cur_corners)

            cur_bs_corners = np.stack(cur_bs_corners, axis=0)
            all_corners.append(cur_bs_corners)
        assert len(all_corners) == batch_size  
        return all_corners
    def my_compute_box_3d(self, center, size, heading_angle):
        R = self.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)
    
    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def restore_aug(self, pred_bboxes, img_metas):
        pred_bboxes_restore=pred_bboxes.copy()
        batch_size = len(pred_bboxes)
        for bs_ind in range(batch_size):
            pred_bboxes_restore[bs_ind] = pred_bboxes_restore[bs_ind].detach().cpu().numpy()
        assert len(pred_bboxes_restore) == batch_size
        return pred_bboxes_restore
    
    def compute_all_bbox_corners_2d(self, pred_bboxes_corners,calib_Rtilt, calib_K):

        batch_size = len(pred_bboxes_corners)
        # batch_size, box_num, _, _ = pred_bboxes_corners.shape

        all_corners_2d = []
        for bs_ind in range(batch_size):
			
            cur_calib_Rtilt = calib_Rtilt[bs_ind,:,:]
            cur_calib_K = calib_K[bs_ind, :, :]

            cur_batch_corners_2d = []
            box_num = pred_bboxes_corners[bs_ind].shape[0]
            for box_ind in range(box_num):
                cur_corners_3d = pred_bboxes_corners[bs_ind][box_ind, :, :]

                cur_corners_2d = self.proj_pointcloud_into_image(cur_corners_3d, cur_calib_Rtilt, cur_calib_K)
                cur_batch_corners_2d.append(cur_corners_2d)

            cur_batch_corners_2d = np.stack(cur_batch_corners_2d, axis=0)
            all_corners_2d.append(cur_batch_corners_2d)
        # all_corners_2d = np.stack(all_corners_2d, axis=0)
        assert len(all_corners_2d) == batch_size
        return all_corners_2d
    
    def crop_patches(self, image, corners_2d, corners_3d, point_cloud, pred_bboxes=None):
        def in_hull(p, hull):
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            return hull.find_simplex(p) >= 0
    
        def extract_pc_in_box3d(pc, box3d):
            """pc: (N,3), box3d: (8,3)"""
            box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
            return pc[box3d_roi_inds, :], box3d_roi_inds

        def is_valid_patch(top_left, down_right, cur_corner_3d, cur_point_cloud):
            patch_ori_size = down_right - top_left
            patch_ori_size = patch_ori_size[0] * patch_ori_size[1]

            if np.isnan(top_left[0]) or np.isnan(top_left[1]) or np.isnan(down_right[0]) or np.isnan(down_right[1]):
                return False, np.array([0,0]), np.array([10,10])

            if top_left[0] < 0:
                top_left[0] = 0

            if top_left[1] < 0:
                top_left[1] = 0

            if top_left[0] >= img_width:
                return False, np.array([0,0]), np.array([10,10])

            if top_left[1] >= img_height:
                return False, np.array([0,0]), np.array([10,10])

            if down_right[0] > img_width:
                down_right[0] = img_width - 1

            if down_right[1] > img_height:
                down_right[1] = img_height - 1

            if down_right[0] <= 0:
                return False, np.array([0,0]), np.array([10,10])

            if down_right[1] <= 0:
                return False, np.array([0,0]), np.array([10,10])

            patch_fixed_size = down_right - top_left

            if patch_fixed_size[0] < 10 or patch_fixed_size[1] < 10:
                return False, np.array([0,0]), np.array([10,10])

            patch_fixed_size = patch_fixed_size[0] * patch_fixed_size[1]

            if patch_fixed_size/patch_ori_size < 0.8:
                return False, top_left, down_right

			# omit if there is no points in the bounding box
            cur_corner_3d = corners_3d[bs_ind][box_ind, :, :]
            pc, _ = extract_pc_in_box3d(cur_point_cloud, cur_corner_3d)
            if pc.shape[0] < 100:
                return False, top_left, down_right
            return True, top_left, down_right

        # batch_size, box_num, _, _ = corners_2d.shape
        batch_size = len(corners_2d)
        all_patch = []

        # all_valid = np.zeros([batch_size, box_num], dtype=np.bool)
        all_valid = []
        for bs_ind in range(batch_size):
            cur_bs_img = image[bs_ind, ...]
            # img_width = image_size[bs_ind, 0]
            # img_height = image_size[bs_ind, 1]

            img_width = 968
            img_height = 1296

            cur_bs_patch = []
            corners_2d_cur_bs = corners_2d[bs_ind]
            box_num = corners_2d_cur_bs.shape[0]
            all_valid_cur_bs = np.zeros([box_num], dtype=np.bool)

            for box_ind in range(box_num):
                cur_corners_2d = corners_2d_cur_bs[box_ind, :, :]
                top_left = np.min(cur_corners_2d, axis=0)
                down_right = np.max(cur_corners_2d, axis=0)
                valid_flag, top_left, down_right = is_valid_patch(top_left, down_right, corners_3d[bs_ind][box_ind, :, :], point_cloud[bs_ind,:,:])
                # all_valid[bs_ind, box_ind] = valid_flag
                all_valid_cur_bs[box_ind] = valid_flag
                top_left = (int(top_left[0]), int(top_left[1]))
                down_right = (int(down_right[0]), int(down_right[1]))

                cur_patch = cur_bs_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :].clone()

                cur_patch = self.img_preprocess(cur_patch)
				
                cur_bs_patch.append(cur_patch)
            all_valid.append(all_valid_cur_bs)
            cur_bs_patch = torch.stack(cur_bs_patch, dim=0)
            cur_bs_patch = cur_bs_patch.permute(0,3,1,2)
            all_patch.append(cur_bs_patch)
        # all_patch = torch.stack(all_patch, dim=0)

        # all_patch = all_patch.permute(0,1,4,2,3)
        assert len(all_patch) == batch_size
        return all_patch, all_valid
    
    def crop_per_patch(self, image, corners_2d, corners_3d, point_cloud, pred_bboxes=None):
        def in_hull(p, hull):
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            return hull.find_simplex(p) >= 0
    
        def extract_pc_in_box3d(pc, box3d):
            """pc: (N,3), box3d: (8,3)"""
            box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
            return pc[box3d_roi_inds, :], box3d_roi_inds

        def is_valid_patch(top_left, down_right, cur_corner_3d, cur_point_cloud):
            patch_ori_size = down_right - top_left
            patch_ori_size = patch_ori_size[0] * patch_ori_size[1]

            if np.isnan(top_left[0]) or np.isnan(top_left[1]) or np.isnan(down_right[0]) or np.isnan(down_right[1]):
                return False, np.array([0,0]), np.array([10,10])

            if top_left[0] < 0:
                top_left[0] = 0

            if top_left[1] < 0:
                top_left[1] = 0

            if top_left[0] >= img_width:
                return False, np.array([0,0]), np.array([10,10])

            if top_left[1] >= img_height:
                return False, np.array([0,0]), np.array([10,10])

            if down_right[0] > img_width:
                down_right[0] = img_width - 1

            if down_right[1] > img_height:
                down_right[1] = img_height - 1

            if down_right[0] <= 0:
                return False, np.array([0,0]), np.array([10,10])

            if down_right[1] <= 0:
                return False, np.array([0,0]), np.array([10,10])

            patch_fixed_size = down_right - top_left

            if patch_fixed_size[0] < 10 or patch_fixed_size[1] < 10:
                return False, np.array([0,0]), np.array([10,10])

            patch_fixed_size = patch_fixed_size[0] * patch_fixed_size[1]

            if patch_fixed_size/patch_ori_size < 0.8:
                return False, top_left, down_right

			# omit if there is no points in the bounding box
            cur_corner_3d = corners_3d
            pc, _ = extract_pc_in_box3d(cur_point_cloud, cur_corner_3d)
            if pc.shape[0] < 100:
                return False, top_left, down_right
            return True, top_left, down_right

        cur_img = image
        img_width = 968
        img_height = 1296
        cur_corners_2d = corners_2d
        top_left = np.min(cur_corners_2d, axis=0)
        down_right = np.max(cur_corners_2d, axis=0)

        valid_flag, top_left, down_right = is_valid_patch(top_left, down_right, corners_3d, point_cloud)
        top_left = (int(top_left[0]), int(top_left[1]))
        down_right = (int(down_right[0]), int(down_right[1]))
        cur_patch = cur_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :].clone()
        cur_patch = self.img_preprocess(cur_patch)
        return cur_patch, valid_flag
            
    def img_preprocess(self, img):
		# clip normalize
        def resize(img, size):
            img_h, img_w, _ = img.shape
            if img_h > img_w:
                new_w = int(img_w * size / img_h)
                new_h = size
            else:
                new_w = size
                new_h = int(img_h * size / img_w)

            dsize = (new_h, new_w)
            transform = T.Resize(dsize, interpolation=BICUBIC)
            img = img.permute([2,0,1])
            if img.shape[1] != 0 and img.shape[2] != 0 :
                if new_h != 0 and new_w != 0 :
                    img = transform(img)
                else:
                    # img = torch.zeros(([3,224,224]),dtype=img.dtype,device=img.device)
                    img = torch.zeros(([3,344,344]),dtype=img.dtype,device=img.device)
            else:
                img = torch.zeros(([3,344,344]),dtype=img.dtype,device=img.device)
                # img = torch.zeros(([3,224,224]),dtype=img.dtype,device=img.device)
            img = img.permute([1,2,0])

            return img
        def center_crop(img, dim):
            """Returns center cropped image
            Args:
            img: image to be center cropped
            dim: dimensions (width, height) to be cropped
            """
            width, height = img.shape[1], img.shape[0]

            # process crop width and height for max available dimension
            crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
            crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
            mid_x, mid_y = int(width/2), int(height/2)
            cw2, ch2 = int(crop_width/2), int(crop_height/2)
            crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
            return crop_img
        def padding(img, dim):
            rst_img = torch.zeros([dim[0], dim[1], 3], dtype=img.dtype, device=img.device)
            h, w, _ = img.shape

            top_left = (np.array(dim)/2 - np.array([h,w])/2).astype(np.int)
            down_right = (top_left + np.array([h,w])).astype(np.int)

            rst_img[top_left[0]:down_right[0], top_left[1]:down_right[1], :] = img.clone()
            return rst_img
        
        # ####"ViT-B/32"
        # img = resize(img, 224)
        # img = padding(img, [224,224])
        ###'ViT-L/14@336px'
        img = resize(img, 344)       
        img = padding(img, [344,344])
        return img

    def collect_matched_query_feat(self, pc_query_feat, gt_labels_3d, assignments=None):

        # query_num, batch_size, feat_dim = pc_query_feat.shape
        batch_size = len(pc_query_feat)

        all_matched_query = []
        all_matched_label = []
        all_matched_prob = []


        for cur_bs in range(batch_size):
            all_matched_query.append(pc_query_feat[cur_bs])
            # all_matched_label.append(gt_labels_3d[cur_bs][assignments[cur_bs]['pred2gt_ind']])
            # all_matched_prob.append(torch.ones([gt_labels_3d[cur_bs][assignments[cur_bs]['pred2gt_ind']].shape[0]], device=gt_labels_3d[cur_bs].device))

        pc_feat_label={"pc_feat": all_matched_query,
                            "pc_label": all_matched_label,
                            "pc_prob": all_matched_prob,
                            }
        return pc_feat_label
    
    def classify(self, image_features, text_features, verbose=False, img_label=None):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logit_scale = logit_scale_.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        probs_ = logits_per_image.softmax(dim=-1)
        probs, rst = torch.max(probs_, dim=1)

        if verbose:
            return probs, rst, probs_, logits_per_image
        else:
            return probs, rst
        
    def dtcc_pc_img_text(self, pair_img_output, pc_output, text_output):
        assert torch.sum(pair_img_output["pair_img_label"] - pc_output["pc_label"]) == 0
        assert torch.sum(pair_img_output["pair_img_prob"] - pc_output["pc_prob"]) == 0 
        class_num = 365 + 4

        pair_img_feat = pair_img_output["pair_img_feat"]
        pair_img_label = pair_img_output["pair_img_label"]
        pair_img_prob = pair_img_output["pair_img_prob"]
        pair_img_label = torch.unsqueeze(pair_img_label, dim=1)
        pair_img_prob = torch.unsqueeze(pair_img_prob, dim=1)
        pair_img_objs = torch.cat([pair_img_feat, pair_img_label, pair_img_prob], dim=1)

        # only keep valid class and prob > average
        valid_ind = torch.where(pair_img_objs[:, -2] < class_num)[0]
        pair_img_objs = pair_img_objs[valid_ind]

        #prepare pc branch
        pc_feat = pc_output["pc_feat"]
        pc_label = pc_output["pc_label"]
        pc_prob = pc_output["pc_prob"]
        pc_label = torch.unsqueeze(pc_label, dim=1)
        pc_prob = torch.unsqueeze(pc_prob, dim=1)
        pc_objs = torch.cat([pc_feat, pc_label, pc_prob], dim=1)

        valid_ind = torch.where(pc_objs[:, -2] < class_num)[0]
        pc_objs = pc_objs[valid_ind]

        #prepare text branch
        text_feat = text_output["text_feat"]
        text_label = text_output["text_label"]
        text_label = torch.unsqueeze(text_label, dim=1)
        text_prob = torch.ones_like(text_label)
        text_objs = torch.cat([text_feat, text_label, text_prob], dim=1)

        # print(pc_objs.shape)

        unique_text_cls = torch.unique(pc_objs[:,-2].detach()).long()
        unique_text_objs = text_objs[unique_text_cls,:]

        rand_text_cls = random.sample(range(0,class_num), 20)
        rand_text_objs = text_objs[rand_text_cls,:]
        text_objs = torch.cat([unique_text_objs, rand_text_objs], dim=0)


        dtcc_group_1 = torch.cat([text_objs, pc_objs], dim=0)
        dtcc_group_2 = torch.cat([pc_objs, pair_img_objs], dim=0)
            
            
        dtcc_loss_1 = self.DTCC_loss(dtcc_group_1)
        dtcc_loss_2 = self.DTCC_loss(dtcc_group_2)

        return dtcc_loss_1, dtcc_loss_2
    def cal_sim(self, z_i,z_j,temperature):
        z_i = z_i / z_i.norm(dim=len(z_i.shape)-1, keepdim=True)
        z_j = z_j / z_j.norm(dim=len(z_j.shape)-1, keepdim=True)
        return z_i @ z_j.t() / temperature
    def DTCC_loss(self, objs, temperature=0.1):
        device = objs.device
        dtcc_loss = torch.tensor(0,device=device,dtype=torch.float)
        criterion = nn.CrossEntropyLoss(reduction="mean").to(device=device)
            
        valid_obj_cnt = 1
        for obj_ind in range(objs.shape[0]):
            obj = objs[obj_ind,:]

            obj_feature = obj[:-2]
            obj_cls = obj[-2]
            obj_score = obj[-1]
            neg_objs_inds = torch.where(objs[:,-2]!=obj_cls)[0]
                    
            if len(neg_objs_inds) > 0:
                neg_objs = objs[neg_objs_inds,:]
                neg_loss = self.cal_sim(obj_feature,neg_objs[:,:-2],temperature).unsqueeze(0)
            else:
                continue
    	
            pos_objs_inds = torch.where(objs[:,-2]==obj_cls)[0]
            pos_objs_inds = [i for i in pos_objs_inds if i!= obj_ind]		# remove itself
 
            if len(pos_objs_inds) > 0:
                pos_loss = self.cal_sim(obj_feature,objs[pos_objs_inds,:-2],temperature).unsqueeze(0).t()
            else:
                pos_loss = torch.tensor([1/temperature],device=device,dtype=torch.float).unsqueeze(0)
                valid_obj_cnt -= 1
	
            logits = torch.cat([pos_loss,neg_loss.repeat(pos_loss.shape[0],1)],dim=1)
            labels = torch.zeros(logits.shape[0], device=device,dtype=torch.long)
                    
            cur_loss = criterion(logits,labels)
                    #print(cur_loss)
            dtcc_loss += cur_loss
            valid_obj_cnt += 1
    
        dtcc_loss /= valid_obj_cnt
        return dtcc_loss
    
    def proj_pointcloud_into_image(self, xyz, pose, calib_K):
        pose = pose.detach().cpu().numpy()
        calib_K = calib_K.detach().cpu().numpy()

        padding = np.ones([xyz.shape[0], 1])
        xyzp = np.concatenate([xyz, padding], axis=1)
        xyz = (pose @ xyzp.T)[:3,:].T

        intrinsic_image = calib_K[:3,:3]
        xyz_uniform = xyz/xyz[:,2:3]
        xyz_uniform = xyz_uniform.T

        uv = intrinsic_image @ xyz_uniform

        uv /= uv[2:3, :]
        uv = np.around(uv).astype(np.int)

        uv = uv.T

        return uv[:,:2]

    def classify_pc(self, pc_query_feat, text_feat, text_num):
        
        # query_num, batch_size, feat_dim = pc_query_feat.shape
        batch_size = 1
        query_num,feat_dim = pc_query_feat.shape
        pc_query_feat = pc_query_feat.reshape([-1, feat_dim])

        pc_all_porb, pc_all_label, pc_all_porb_ori, _ = self.classify(pc_query_feat.half(), text_feat, verbose=True)

        # pc_all_label = pc_all_label.reshape([query_num, batch_size])
        # pc_all_porb = pc_all_porb.reshape([query_num, batch_size])
        # pc_all_porb_ori = pc_all_porb_ori.reshape([query_num, batch_size, -1])

        # pc_all_logits = torch.zeros([query_num, batch_size, text_num+1], device=pc_all_porb_ori.device)
        pc_all_logits = torch.zeros([query_num, text_num+1], device=pc_all_porb_ori.device)
        pc_all_logits[..., :text_num] = torch.log(pc_all_porb_ori[...,:text_num])
        pc_all_logits[..., text_num] = torch.log(torch.sum(pc_all_porb_ori[...,text_num:], dim=-1))

        # pc_all_logits = pc_all_logits.permute(1,0,2)
        # pc_all_porb = pc_all_porb.permute(1,0)
        # pc_all_porb_ori = pc_all_porb_ori[:,:,:text_num].permute(1,0,2)
        # pc_all_label = pc_all_label.permute(1,0)
        pc_all_porb_ori = pc_all_porb_ori[:,:text_num]
        return pc_all_logits, pc_all_porb, pc_all_porb_ori, pc_all_label

    def build_img_encoder(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        # encoder, preprocess = clip.load("ViT-B/32", device=device)
        #encoder, preprocess = clip.load("RN50", device=device)
        encoder, preprocess = clip.load("ViT-L/14@336px", device=device)
        return encoder

    def batch_encode_text(self, text):
        batch_size = 20

        text_num = text.shape[0]
        cur_start = 0
        cur_end = 0

        all_text_feats = []
        while cur_end < text_num:
			# print(cur_end)
            cur_start = cur_end
            cur_end += batch_size
            if cur_end >= text_num:
                cur_end = text_num
			
            cur_text = text[cur_start:cur_end,:]
            cur_text_feats = self.img_model.encode_text(cur_text).detach()
            all_text_feats.append(cur_text_feats)

        all_text_feats = torch.cat(all_text_feats, dim=0)
		# print(all_text_feats.shape)
        return all_text_feats
    
    def bbox_select_train(self, preds_bboxes, gt_bboxes_3d):
        second_pred_bboxes, assigns = [], []
        for bs in range(len(preds_bboxes)):
            bbox_pred = preds_bboxes[bs][0]
            bbox_gt = gt_bboxes_3d[bs]
            mask = []
            final_costs = []
            for n in range(bbox_gt.shape[0]):
                gt = bbox_gt[n].unsqueeze(0).expand(bbox_pred.shape[0], 7)
                ####iou
                pred_to_targets_iou = diff_iou_rotated_3d(bbox_pred.unsqueeze(0), gt.unsqueeze(0)).squeeze(0).detach()
                ####center_distance
                center_distance =  torch.norm((bbox_pred[:,:3] - gt[:,:3]),dim=1).detach()
                final_cost = 3 * ( - pred_to_targets_iou) + 5 * center_distance
                final_costs.append(final_cost)
            final_costs = torch.stack(final_costs,dim=1)
            assign = linear_sum_assignment(final_costs.cpu())
            assign = [torch.from_numpy(x).long().to(bbox_pred.device) for x in assign]
            second_pred_bboxes.append(bbox_pred[assign[0]])
            assigns.append(assign[1])
        return second_pred_bboxes, assigns

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
   
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot
