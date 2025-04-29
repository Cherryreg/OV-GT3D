import numpy as np
import os
import cv2
from tqdm import tqdm
import multiprocessing
import pdb

def count_point_inside_box(bbox, points, axis_alignment_matrix, aligned):
    center = bbox[:3]
    box_width = bbox[3:6]
    xyz = points[:, :3]
    if aligned:
        padding = np.expand_dims(np.ones(xyz.shape[0]), axis=1)
        xyz_pad = np.concatenate([xyz, padding], axis=1)
        xyz_aligned = axis_alignment_matrix @ xyz_pad.T
        xyz_aligned = xyz_aligned.T[:, :3]
        xyz = xyz_aligned
        #np.savetxt('xyz.txt', xyz, fmt='%.3f')
        #exit()

    xyz = xyz - np.expand_dims(center, axis=0)

    mask = (abs(xyz) <= box_width/2).all(axis=1)
    inside_count = np.count_nonzero(mask)
    return inside_count

def get_gt_image_from_pc_cur_scan(bboxes, points, pose, height, width, axis_alignment_matrix, aligned):
    bbox_centers = bboxes[:, :3]

    bbox_centers_uniform = bbox_centers.T

    pose_inv = np.linalg.inv(pose)
    padding = np.ones(bbox_centers_uniform.shape[1])
    padding = np.expand_dims(padding, axis=0)
    bbox_centers_uniform = np.concatenate([bbox_centers_uniform, padding], axis=0)
    uv_before_intrinsic = pose_inv @ bbox_centers_uniform

    
    uv_before_intrinsic = uv_before_intrinsic[:3,:]
    uv = uv_before_intrinsic

    #uv = intrinsic_image @ uv_before_intrinsic

    uv /= uv[2:3, :]
    uv = np.around(uv).astype(int)
    uv = uv.T

    num_point_in_bbox = np.zeros(bboxes.shape[0]) - 1

    #valid_mask = (0 <= uv[:, 0]) & (uv[:, 0] <= width) & (0 <= uv[:, 1]) & (uv[:, 1] <= height)
    #index_valid = np.where(valid_mask)[0]
    #num_point_in_bbox[~index_valid] = -1
    index_valid = range(0, bboxes.shape[0])
    #TODO: 可不可以扔掉for
    for index in index_valid:
        bbox = bboxes[index, :]
        num_point_in_bbox[index] = count_point_inside_box(bbox, points, axis_alignment_matrix, aligned)
        #print('bbox', index, bbox[-1], 'is in sight!')
        #print(num_point_in_bbox[index], " pixels")
    
    return num_point_in_bbox

def convert_from_uvd(u, v, d, intr, pose):
    # u is width index, v is height index
    depth_scale = 1000
    z = d/depth_scale

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    uv = np.concatenate([u,v,padding], axis=0)
    #将每个点投影回对齐的坐标系下
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) * np.expand_dims(z,axis=0)
    xyz_local = xyz.copy()
    
    xyz = np.concatenate([xyz,padding], axis=0)

    xyz = pose @ xyz
    
    xyz[:3,:] /= xyz[3,:] 

    return xyz[:3, :].T, xyz_local.T


def get_gt_image_from_pc(cur_scan):
    scan_name_index = cur_scan["scan_name_index"]
    scan_num = cur_scan["scan_num"]
    print("%d/%d"%(scan_name_index+1, scan_num))

    path_dict = cur_scan["path_dict"]
    aligned = cur_scan['aligned']
    scan_name = cur_scan["scan_name"]


    DATA_PATH = path_dict["DATA_PATH"]
    DATA_PC_PATH = path_dict["DATA_PC_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    BBOX_PATH = path_dict["BBOX_PATH"]
    RGB_PATH = path_dict["RGB_PATH"]
    DEPTH_PATH = path_dict["DEPTH_PATH"]
    POSE_PATH = path_dict["POSE_PATH"]

    scan_name = scan_name.strip("\n")
    scan_path = os.path.join(DATA_PATH,scan_name)
    scan_pc_path = os.path.join(DATA_PC_PATH,scan_name)
    path_dict["scan_path"] = scan_path
    path_dict["scan_pc_path"] = scan_pc_path
    
    scan_folder_path = os.path.join(TARGET_DIR)
    os.makedirs(scan_folder_path, exist_ok=True)

    POSE_txt_list = sorted(os.listdir(os.path.join(scan_path,POSE_PATH)))
    rgb_map_list = sorted(os.listdir(os.path.join(scan_path,RGB_PATH)))
    depth_map_list = sorted(os.listdir(os.path.join(scan_path,DEPTH_PATH)))
    poses = [np.loadtxt(os.path.join(scan_path,POSE_PATH, i)).reshape(4,4) for i in POSE_txt_list]
    

    intrinsic_depth = np.loadtxt(os.path.join(scan_path, 'intrinsics_depth.txt')).reshape(4,4)
    intrinsic_image = np.loadtxt(os.path.join(scan_path, 'intrinsics_color.txt')).reshape(4,4)

    pc_info_filename = os.path.join(DATA_PC_PATH, scan_name, "%s.txt"%(scan_name))
    with open(pc_info_filename, 'r') as pc_info_file:
        for line in pc_info_file:
            key, value = line.strip().split(' = ')
            if key == 'axisAlignment':    
                #print(value)            
                axis_alignment_matrix = np.array(value.split(' '), dtype='float').reshape(4,4)
                break
    if aligned:
        bboxes_filename = os.path.join(BBOX_PATH, '%s_aligned_bbox.npy')%(scan_name)
    else:
        bboxes_filename = os.path.join(BBOX_PATH, '%s_unaligned_bbox.npy')%(scan_name)
    bboxes = np.load(bboxes_filename)

    num_point_in_bbox_list = []
    for i, (rgb_map_name, depth_map_name, pose) in enumerate(zip(rgb_map_list, depth_map_list, poses)):
        #print(scan_name, "%i/%i"%(i, len(rgb_map_list)), rgb_map_name)
        depth_map = cv2.imread(os.path.join(scan_path,DEPTH_PATH,depth_map_name),-1)
        color_map = cv2.imread(os.path.join(scan_path,RGB_PATH,rgb_map_name))
        color_map = cv2.cvtColor(color_map,cv2.COLOR_BGR2RGB)

        # convert depth map to point cloud
        height, width = depth_map.shape  
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)

        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]

        xyz, xyz_local = convert_from_uvd(ww_ind, hh_ind, depth_map, intrinsic_depth, pose)

        num_point_in_bbox = get_gt_image_from_pc_cur_scan(bboxes, xyz, pose, height, width, axis_alignment_matrix, aligned)

        num_point_in_bbox_list.append(num_point_in_bbox)

    num_point_in_bbox_list = np.array(num_point_in_bbox_list)
    # np.savetxt('pts_in_box.txt', num_point_in_bbox_list, fmt='%.0f')
    max_point_in_bbox = np.max(num_point_in_bbox_list, axis=0)
    max_point_in_bbox_filtered = np.where(max_point_in_bbox <= 100, -1, max_point_in_bbox)
    bbox_not_in_image = np.where(np.equal(max_point_in_bbox_filtered, -1))[0]
    
    optimal_image = np.argmax(np.array(num_point_in_bbox_list), axis=0)

    optimal_image[bbox_not_in_image] = len(rgb_map_list)
    rgb_map_list.append('999999')
    
    # optimal_image_str =  np.array(['{:06d}'.format(x * 100) if x != -1 else '999999' for x in optimal_image])
    optimal_image_str =  np.array(rgb_map_list)[optimal_image]
    optimal_image_filename = os.path.join(TARGET_DIR, "%s_gtbox_image.npy"%(scan_name))
    np.save(optimal_image_filename, optimal_image_str)

def main():
    DATA_PATH = "./scannet_25k/scannet_frames_25k" # Replace it with the path to scannet_frames_25k
    DATA_PC_PATH = "./scannet/scans" # Replace it with the path to 3D
    TARGET_DIR = "./scannet/scannet_OIS3D/gt_images" # Replace it with the path to output path\
    BBOX_PATH = "./scannet/scannet_OIS3D/scannet_instance_data"  #Replace it with the path to OWP_scannet_all/scannet_instance_data
    DATA_METADATA_PREFIX = "./scannet/meta_data"
    RGB_PATH = "./color"
    DEPTH_PATH = "./depth"
    INSTANCE_PATH = "./instance"
    LABEL_PATH = "./label"
    POSE_PATH = "./pose"

    aligned = True #align pointcloud with axis or not

    path_dict = {"DATA_PATH": DATA_PATH,
                "TARGET_DIR": TARGET_DIR,
                "RGB_PATH": RGB_PATH,
                "DEPTH_PATH": DEPTH_PATH,
                "INSTANCE_PATH": INSTANCE_PATH,
                "LABEL_PATH": LABEL_PATH,
                "POSE_PATH": POSE_PATH,  
                "DATA_PC_PATH": DATA_PC_PATH ,
                "BBOX_PATH": BBOX_PATH   
                }

    os.makedirs(TARGET_DIR,exist_ok=True)
    for split in ['train', 'val']:
        print(split)
        f = open(os.path.join(DATA_METADATA_PREFIX, "scannetv2_%s.txt"%(split)))
        scan_name_list = sorted(f.readlines())

        multi_process_parameter = []
        for scan_index, scan_name in enumerate(scan_name_list):
            cur_parameter = {}
            cur_parameter["scan_name_index"] = scan_index
            cur_parameter["scan_name"] = scan_name
            cur_parameter["path_dict"] = path_dict
            cur_parameter["scan_num"] = len(scan_name_list)
            cur_parameter["aligned"] = aligned
            multi_process_parameter.append(cur_parameter)
            #pool.map(get_gt_image_from_pc, multi_process_parameter)
            #cProfile.runctx('process_cur_scan_pc(cur_parameter)', {'cur_parameter'     : cur_parameter, 'process_cur_scan_pc' : process_cur_scan_pc}, {})
            get_gt_image_from_pc(cur_parameter)

if __name__ == "__main__":
    main()

