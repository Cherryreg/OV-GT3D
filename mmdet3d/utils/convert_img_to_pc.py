import numpy as np
import cv2
import os
import open3d as o3d
import os
'''
def img2pc(scan_name='scene0010_01', img_name='000000'):
    def get_color_label(xyz, intrinsic_image, rgb):
        height, width, ins_num = rgb.shape
        intrinsic_image = intrinsic_image[:3,:3]

        xyz_uniform = xyz/xyz[:,2:3]
        xyz_uniform = xyz_uniform.T

        uv = intrinsic_image @ xyz_uniform

        uv /= uv[2:3, :]
        uv = np.around(uv).astype(np.int)
        uv = uv.T

        uv[:, 0] = np.clip(uv[:, 0], 0, width-1)
        uv[:, 1] = np.clip(uv[:, 1], 0, height-1)

        uv_ind = uv[:, 1]*width + uv[:, 0]

        pc_rgb = np.take_along_axis(rgb.reshape([-1,3]), np.expand_dims(uv_ind, axis=1), axis=0)
        return pc_rgb

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
    scannet_25k_path = '/home/yifei/hyf/data/scannet_frames_25k'
    depth_map = cv2.imread(os.path.join(scannet_25k_path, scan_name,'depth' ,img_name+'.png'),-1)
    color_map = cv2.imread(os.path.join(scannet_25k_path, scan_name,'color' ,img_name+'.jpg'))
    color_map = cv2.cvtColor(color_map,cv2.COLOR_BGR2RGB)
    intrinsic_depth = np.loadtxt(os.path.join(scannet_25k_path, scan_name, 'intrinsics_depth.txt')).reshape(4,4)
    intrinsic_image = np.loadtxt(os.path.join(scannet_25k_path, scan_name, 'intrinsics_color.txt')).reshape(4,4)
    pose = np.loadtxt(os.path.join(scannet_25k_path, scan_name,'pose', img_name+'.txt')).reshape(4,4)
    
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
    xyz, xyz_local = convert_from_uvd(ww_ind, hh_ind, depth_map,intrinsic_depth, pose)
    rgb = get_color_label(xyz_local, intrinsic_image, color_map)
    return np.concatenate((xyz, rgb), axis=1)

scan_name='scene0010_01'
img_name='000100'
points = img2pc(scan_name, img_name)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(points[:,3:]/255)

file_path = os.path.join('/home/yifei/hyf/data/ScanNet_processed/scannet_frames_25k_val', scan_name + '_' + img_name + '_pose.txt')
align_matrix = np.loadtxt(file_path)

pcd.transform(np.linalg.inv(align_matrix))

o3d.io.write_point_cloud("/home/yifei/hyf/fcaf3d/output_file_sv/colored_pc/" + scan_name + "_" + img_name + ".ply", pcd)
'''
scene_name = 'scene0487_00_000000'
file_prefix = '/home/yifei/hyf/data/ScanNet_processed/scannet_frames_25k_val'
file_name= os.path.join(file_prefix, scene_name + '_pc.npy')
points = np.load(file_name)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:,:3])
pcd.colors = o3d.utility.Vector3dVector(points[:,3:]/255)

o3d.io.write_point_cloud("/home/yifei/hyf/fcaf3d/output_file_sv/colored_pc/" + scene_name + ".ply", pcd)

